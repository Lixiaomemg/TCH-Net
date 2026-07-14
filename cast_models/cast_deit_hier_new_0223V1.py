"""Define CAST model for classification following DeiT convention.

Modified from:
    https://github.com/facebookresearch/moco-v3/blob/main/vits.py
    https://github.com/facebookresearch/deit/blob/main/models.py
"""
"""

"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial, reduce
from operator import mul

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import PatchEmbed
from timm.models.layers import trunc_normal_

from cast_models.utils import segment_mean_nd
from cast_models.graph_pool import GraphPooling
from cast_models.modules import Pooling, ConvStem

__all__ = [
    'cast_small',
    'cast_small_deep',
    'cast_base',
    'cast_base_deep',
]


    
NEW_EMBED_DIM = 384     

class GeneralMultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qk_dim=None, v_dim=None, attn_drop=0., proj_drop=0., bias=False):
        super().__init__()
        self.num_heads = num_heads
        C = dim 
        qk_dim = qk_dim if qk_dim is not None else C 
        v_dim = v_dim if v_dim is not None else C
        self.scale = (qk_dim // num_heads) ** -0.5

        self.q_proj = nn.Linear(dim, qk_dim, bias=bias)
        self.k_proj = nn.Linear(dim, qk_dim, bias=bias)
        self.v_proj = nn.Linear(dim, v_dim, bias=bias)
        
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(v_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, q_feat, k_feat, v_feat, mask=None):
        """
        Args:
          q_feat: Query 特征。形状 [B, N_q, D]
          k_feat: Key 特征。形状 [B, N_k, D]
          v_feat: Value 特征。形状 [B, N_k, D]
          mask:   注意力掩码。形状 [B, N_q, N_k] 或 [N_q, N_k]
        """
        B = q_feat.shape[0]
        q = self.q_proj(self.norm_q(q_feat)) # [B, N_q, C_qk]
        k = self.k_proj(self.norm_kv(k_feat)) # [B, N_k, C_qk]
        v = self.v_proj(self.norm_kv(v_feat)) # [B, N_k, C_v]
        
        C_qk = q.shape[-1]
        C_v = v.shape[-1]

        # [B, N, C] -> [B, N, H, C/H] -> [B, H, N, C/H]
        q = q.reshape(B, -1, self.num_heads, C_qk // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, C_qk // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, C_v // self.num_heads).permute(0, 2, 1, 3)

        # 1. Attention Score
        # q @ k.transpose: [B, H, N_q, C/H] @ [B, H, C/H, N_k] -> [B, H, N_q, N_k]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 3. Softmax
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 4. Attention * V
        # attn @ v: [B, H, N_q, N_k] @ [B, H, N_k, C/H] -> [B, H, N_q, C/H]
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C_v) 
        
        # 5. [B, H, N_q, C/H] -> [B, N_q, H, C/H] -> [B, N_q, C_v]

        # 6. 
        x = self.proj(x)
        x = self.proj_drop(x)

        if mask is not None:
            x = x.max(dim=1, keepdim=True)[0] # [B, 1, C]
        return x

        
def aggregate_prototypes(fine_prototypes, hierarchy_map):
    """
    根据层级映射表，将细粒度原型聚合为粗粒度原型。
    
    Args:
        fine_prototypes (torch.Tensor): 细粒度原型 [N_fine, D]
        hierarchy_map (dict): {coarse_id: [fine_id_1, fine_id_2, ...]}
    
    Returns:
        torch.Tensor: 聚合后的粗粒度原型 [N_coarse, D]
    """
    device = fine_prototypes.device
    dim = fine_prototypes.shape[-1]
    coarse_protos = []
    
    # 确保 coarse_id 遍历是连续的
    coarse_ids = sorted(hierarchy_map.keys()) 
    
    for c_idx in coarse_ids:
        child_indices = hierarchy_map.get(c_idx, [])
        if len(child_indices) > 0:
            child_indices_tensor = torch.tensor(child_indices, device=device)
            child_protos = fine_prototypes[child_indices_tensor].mean(dim=0)
        else:
            # 如果某个粗粒度类别没有子类别，则初始化为零向量（或平均细粒度原型）
            child_protos = torch.zeros(dim, device=device)
        coarse_protos.append(child_protos)
    return torch.stack(coarse_protos)


class CAST(VisionTransformer):
    def __init__(self, nb_classes, hierarchy_maps = None,clip_embeds=None, *args, **kwargs):
        depths = kwargs['depth']
        num_clusters = kwargs.pop('num_clusters', [64, 32, 16, 8])
        kwargs['depth'] = sum(kwargs['depth'])
        super().__init__(**kwargs)

        assert self.dist_token is None, 'dist_token is not None.'
        assert self.head_dist is None, 'head_dist is not None.'
 
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

        # --- Point 1: Class Mapping Correction (Strict User Naming) ---
        if len(nb_classes) == 3:
            self.num_classes = nb_classes[0]      # Fine
            self.num_family = nb_classes[1]       # Mid
            self.num_manufacturer = nb_classes[2] # Coarser
            self.num_levels = 3
        elif len(nb_classes) == 2:
            self.num_classes = nb_classes[0]      # Fine
            self.num_family = nb_classes[1]       # Coarser (L2)
            self.num_manufacturer = 0             # Not used
            self.num_levels = 2

        # --- Prototype Initialization and Guidance Modules ---
        D = self.embed_dim
        H = kwargs.get('num_heads', 8)

        # Fine Layer Prototypes (self.num_classes)
        self.fine_prototypes = nn.Parameter(clip_embeds.clone().detach())
        self.fine_refinement_attn = GeneralMultiHeadAttention(D, num_heads=H)

        if self.num_levels == 3:
            mid_initial_protos = aggregate_prototypes(self.fine_prototypes, hierarchy_maps[0])
            self.mid_prototypes = nn.Parameter(mid_initial_protos)
            self.mid_refinement_attn = GeneralMultiHeadAttention(D, num_heads=H)
            self.fine_to_mid_attn = GeneralMultiHeadAttention(D, num_heads=H)
            self.coarser_refinement_attn = GeneralMultiHeadAttention(D, num_heads=H)
        else:
            # Coarser Layer Prototypes (self.num_manufacturer in 3L, or uses num_family in 2L)
            num_coarser_proto = self.num_manufacturer if self.num_levels == 3 else self.num_family
            self.coarser_refinement_attn = GeneralMultiHeadAttention(D, num_heads=H)

        if self.num_levels == 3:
            self.mid_to_coarser_attn = GeneralMultiHeadAttention(D, num_heads=H)
        elif self.num_levels == 2:
            self.fine_to_coarser_attn = GeneralMultiHeadAttention(D, num_heads=H)

        # --- Head Initialization (D for D+D fusion) ---
        # Fine Head (Used: self.head) -> maps to self.num_classes
        num_fine_head = self.num_classes
        self.head = nn.Linear(D, num_fine_head) if num_fine_head > 0 else nn.Identity()
        # Mid Head (Used: self.family_head) -> maps to self.num_family (3L Mid, 2L Coarser)
        self.family_head = nn.Linear(D, self.num_family) if self.num_family > 0 else nn.Identity()
        # Coarser Head (Used: self.manufacturer_head) -> maps to self.num_manufacturer (3L Coarser)
        self.manufacturer_head = nn.Linear(D, self.num_manufacturer) if self.num_manufacturer > 0 else nn.Identity()
        
        # Initialize weights
        self.manufacturer_head.apply(self._init_weights)
        self.family_head.apply(self._init_weights)
        self.head.apply(self._init_weights)

        # Original Block Initialization (Keep)
        cumsum_depth = [0]
        for d in depths:
            cumsum_depth.append(d + cumsum_depth[-1])

        blocks = []
        pools = []
        for ind, depth in enumerate(depths):
            blocks.append(self.blocks[cumsum_depth[ind]:cumsum_depth[ind+1]])
            pool = Pooling(
                pool_block=GraphPooling(
                    num_clusters=num_clusters[ind],
                    d_model=D,
                    l2_normalize_for_fps=False))
            pools.append(pool)

        self.blocks1, self.pool1 = blocks[0], pools[0]
        self.blocks2, self.pool2 = blocks[1], pools[1]
        self.blocks3, self.pool3 = blocks[2], pools[2]
        self.blocks4, self.pool4 = blocks[3], pools[3]


    def _block_operations(self, x, cls_token, x_pad_mask,
                          nn_block, pool_block, norm_block):
        """Wrapper to define operations per block."""
        cls_x = torch.cat([cls_token, x], dim=1)
        cls_x = nn_block(cls_x).type_as(x)
        cls_token, x = cls_x[:, :1, :], cls_x[:, 1:, :]

        cls_token, pool_logit, centroid, pool_pad_mask, pool_inds = (
            pool_block(cls_token, x, x_pad_mask)
        )

        if norm_block is not None:
            out = norm_block(cls_x)[:, 0]
        else:
            out = cls_x[:, 0]

        return (x, cls_token, pool_logit, centroid,
                pool_pad_mask, pool_inds, out)

    # --- Function 1: Segment Token to Prototype Feature Extraction (No Change) ---
    def _get_prototype_feature(self, segment_tokens, layer_prototypes, prototype_attn_module):
        B = segment_tokens.shape[0]
        N_p = layer_prototypes.shape[0]
        D = layer_prototypes.shape[1]
        
        prototypes_kv = layer_prototypes.unsqueeze(0).expand(B, N_p, D)
        segment_query = segment_tokens#.mean(dim=1, keepdim=True) # (B, 1, D)

        prototype_feature = prototype_attn_module(
            q=segment_query, 
            k=prototypes_kv, 
            v=prototypes_kv,
            #residual=segment_query
        ) 
        return prototype_feature # (B, 1, D)

    # --- Function 2: Finer Prototype to Coarser CLS Token Guidance (Safety Check Kept) ---
    def _guide_cls_token(self, finer_prototype_feature, coarser_cls_token, guidance_attn_module):
        q = coarser_cls_token # Expected (B, 1, D)
        
        # B=1 collapse check: If the input loses its batch dim, add it back.
        if q.dim() == 2:
            q = q.unsqueeze(0) 
            
        guided_cls_token = guidance_attn_module(
            q=q,
            k=finer_prototype_feature, # Expected (B, 1, D)
            v=finer_prototype_feature,
            residual=q # Use the 3D-safe Q for residual
        )
        return guided_cls_token # (B, 1, D)

    def get_orthogonality_loss(self,prototypes,tag= "fine"):
        """原型正交约束"""
        protos = F.normalize(prototypes, p=2, dim=1)
        gram = torch.mm(protos, protos.t())
        if tag == "fine":
            eye = torch.eye(self.num_classes, device=protos.device)
        elif tag == "mid":
            eye = torch.eye(self.num_family, device=protos.device)
        else:
            eye = torch.eye(self.num_manufacturer, device=protos.device)
        # 使用 Mean Squared Error
        return ((gram - eye) ** 2).mean()
    
    def forward_features(self, x, y): 
        x = self.patch_embed(x) 
        N, H, W, C = x.shape
        y = y.unsqueeze(1).float()
        y = F.interpolate(y, x.shape[1:3], mode='nearest')
        y = y.squeeze(1).long()  
        # Assuming segment_mean_nd is available
        x = segment_mean_nd(x, y) 
        ones = torch.ones((N, H, W, 1), dtype=x.dtype, device=x.device)
        # Assuming segment_mean_nd is available
        avg_ones = segment_mean_nd(ones, y).squeeze(-1)
        x_padding_mask = avg_ones <= 0.5
        pos_embed = self.pos_embed[:, 1:].view(1, H, W, C).expand(N, -1, -1, -1) 
        pos_embed = segment_mean_nd(pos_embed, y)  
        x = self.pos_drop(x + pos_embed)  
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        cls_token = cls_token + self.pos_embed[:, :1]
        intermediates = {}

        # Block1 (Finest Segment Tokens Centroid 1)
        (block1, cls_token1, pool_logit1, centroid1,
         pool_padding_mask1, pool_inds1, out1) = self._block_operations(
            x, cls_token, x_padding_mask, self.blocks1, self.pool1, None)
        intermediates.update({'cls_token1': cls_token1, 'centroid1': centroid1, 'out1': out1})

        # Block2 -> Fine CLS Token / Fine Prototype Input Segment (L3 Fine)
        (block2, cls_token2, pool_logit2, centroid2,
         pool_padding_mask2, pool_inds2, out2) = self._block_operations(
            centroid1, cls_token1, pool_padding_mask1, self.blocks2, self.pool2, None)
        intermediates.update({'cls_token2': cls_token2, 'centroid2': centroid2, 'out2': out2})

        # Block3 -> Mid CLS Token (L3 Mid) / Fine CLS Token (L2 Fine)
        (block3, cls_token3, pool_logit3, centroid3,
         pool_padding_mask3, pool_inds3, out3) = self._block_operations(
            centroid2, cls_token2, pool_padding_mask2, self.blocks3, self.pool3, None)
        intermediates.update({'cls_token3': cls_token3, 'centroid3': centroid3, 'out3': out3})

        # Block4 -> Coarser CLS Token (L3/L2 Coarser)
        (block4, cls_token4, pool_logit4, centroid4,
         pool_padding_mask4, pool_inds4, out4) = self._block_operations(
            centroid3, cls_token3, pool_padding_mask3, self.blocks4, self.pool4, self.norm)
        out4 = self.pre_logits(out4)
        intermediates.update({'cls_token4': cls_token4, 'centroid4': centroid4, 'out4': out4})

        return intermediates

    def forward(self, x, y):
        ortho_loss = 0
        intermediates = self.forward_features(x, y)
        # proto_loss = torch.tensor(0.0, device=x.device) # Commented out as labels are missing
        B = x.shape[0]
        D = self.embed_dim
        fine_protos_kv = self.fine_prototypes.unsqueeze(0).expand(B, -1, -1)
        if self.num_levels == 3:
            # --- FINE Layer (L3) --- (out2, centroid1)
            fine_proto_f = self.fine_refinement_attn(
                intermediates['centroid2'], fine_protos_kv, fine_protos_kv, mask = True,# =============================fine enhance======================
            ).squeeze(1) # (B, D)
            fine_cls = intermediates['out2'] # (B, D)
            fine_out = self.head(fine_proto_f + fine_cls) # =============================fine Head======================
            ortho_loss += self.get_orthogonality_loss(self.fine_prototypes,tag= "fine")

            # --- MID Layer (L3) --- (out3 guided by fine_proto_f, centroid2)
            mid_protos_kv = self.mid_prototypes.unsqueeze(0).expand(B, -1, -1)
            mid_proto_f = self.mid_refinement_attn(
                intermediates['centroid3'], mid_protos_kv, mid_protos_kv, mask = True,# =============================mid enhance======================
            ).squeeze(1) # (B, D)
            # CLS token (out3) must be unsqueezed to (B, 1, D) for attention Q input
            guided_mid_cls = self.fine_to_mid_attn(
                intermediates['out3'].unsqueeze(1),fine_proto_f.unsqueeze(1),  fine_proto_f.unsqueeze(1)).squeeze(1)
            mid_out = self.family_head(guided_mid_cls +intermediates['out3'])# =====  + mid_proto_f========================mid Head======================
            ortho_loss += self.get_orthogonality_loss(self.mid_prototypes,tag= "mid")

            guided_coarser_cls = self.mid_to_coarser_attn(
                intermediates['out4'].unsqueeze(1),mid_proto_f.unsqueeze(1),  mid_proto_f.unsqueeze(1)
            ).squeeze(1) # (B, D)
            coarser_out = self.manufacturer_head(guided_coarser_cls + intermediates['out4'] )# ======+ coarser_proto_f=======================coarser Head======================
            return fine_out, mid_out, coarser_out ,ortho_loss
    
        elif self.num_levels == 2:
            # --- FINE Layer (L2) --- (out3, centroid2) <-- **FIXED** to match user logic
            # print ("self.fine_prototypes",self.fine_prototypes)
            fine_proto_f = self.fine_refinement_attn(
                        intermediates['centroid3'], fine_protos_kv, fine_protos_kv, mask = True,# =============================fine enhance======================
                        ).squeeze(1) # (B, D)

            fine_cls = intermediates['out3'] # (B, D)
            fine_out = self.head(fine_proto_f + fine_cls) # =============================fine Head======================
            ortho_loss += self.get_orthogonality_loss(self.fine_prototypes,tag= "fine")
            guided_coarser_cls = self.fine_to_coarser_attn(
                        intermediates['out4'].unsqueeze(1),fine_proto_f.unsqueeze(1),  fine_proto_f.unsqueeze(1)
                        ).squeeze(1) # (B, D)
            coarser_out = self.family_head( guided_coarser_cls +intermediates['out4']) # ===== + coarser_proto_f========================mid Head======================

            return fine_out, coarser_out,ortho_loss




@register_model
def cast_small(pretrained=False, **kwargs):
    # minus one ViT block
    model = CAST(
        patch_size=8, embed_dim=384, num_clusters=[64, 32, 16, 8],
        depth=[3, 3, 3, 2], num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def cast_small_deep(pretrained=False, **kwargs):
    # minus one ViT block
    model = CAST(
        patch_size=8, embed_dim=384, num_clusters=[64, 32, 16, 8],
        depth=[6, 3, 3, 3], num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def cast_base(pretrained=False, **kwargs):
    # minus one ViT block
    model = CAST(
        patch_size=8, embed_dim=768, num_clusters=[64, 32, 16, 8],
        depth=[3, 3, 3, 2], num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def cast_base_deep(pretrained=False, **kwargs):
    # minus one ViT block
    model = CAST(
        patch_size=8, embed_dim=768, num_clusters=[64, 32, 16, 8],
        depth=[6, 3, 3, 3], num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model
