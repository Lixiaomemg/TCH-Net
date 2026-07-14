"""Define CAST model for classification following DeiT convention.

Modified from:
    https://github.com/facebookresearch/moco-v3/blob/main/vits.py
    https://github.com/facebookresearch/deit/blob/main/models.py
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


class CAST(VisionTransformer):
    def __init__(self, *args, **kwargs):
        depths = kwargs['depth']
        # These entries do not exist in timm.VisionTransformer.
        # 弹出 num_clusters，作为每个阶段 Graph Pooling 的目标聚类/节点数。
        # 默认值为 [64, 32, 16, 8]，表示四个阶段的目标节点数。
        num_clusters = kwargs.pop('num_clusters', [64, 32, 16, 8])
        # 将所有阶段的深度相加，得到 VisionTransformer 期望的总深度
        # 因为 VisionTransformer 预期一个平坦的块列表。
        kwargs['depth'] = sum(kwargs['depth'])
        # 调用父类 VisionTransformer 的构造函数，初始化基础结构 (如 patch_embed, blocks, norm, cls_token 等)
        super().__init__(**kwargs)

        # 确保不使用 dist_token 和 head_dist，这是原始 DeiT 或其他变体中用于蒸馏的。
        assert self.dist_token is None, 'dist_token is not None.'
        assert self.head_dist is None, 'head_dist is not None.'

        # 计算分块嵌入后的序列长度 (不包括 cls_token)
        num_patches = self.patch_embed.num_patches
        # 重新初始化位置编码 (pos_embed)，以支持后续的分层操作，通常 ViT 的 pos_embed 包含 cls_token 和所有 patch tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        # 使用截断正态分布初始化位置编码
        trunc_normal_(self.pos_embed, std=.02)

        # 分类头，用于将嵌入维度映射到类别数
        self.head = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        # 编码器结构的分阶段设置
        # 计算每个阶段在父类 self.blocks 列表中的起始索引
        cumsum_depth = [0]
        for d in depths:
            cumsum_depth.append(d + cumsum_depth[-1])

        blocks = []# 用于存储每个阶段的 Attention Blocks
        pools = []# 用于存储每个阶段的 Pooling Layers

        # 遍历每个阶段的深度
        for ind, depth in enumerate(depths):
            # 提取 Attention Blocks。从父类 self.blocks 中切片出当前阶段的块
            blocks.append(self.blocks[cumsum_depth[ind]:cumsum_depth[ind+1]])

            # 构建 Pooling layers
            pool = Pooling(
                pool_block=GraphPooling(                # 使用 GraphPooling 机制
                    num_clusters=num_clusters[ind],     # 当前阶段的目标聚类/节点数
                    d_model=kwargs['embed_dim'],        # 嵌入维度
                    l2_normalize_for_fps=False))
            
            # 最后一个 Graph Pooling 不需要，因为它后面没有注意力块，只用于产生最终特征
            # 因此，通常会冻结最后一个 Graph Pooling 层的参数（此处冻结了其内部的全连接层参数）
            if ind == len(depths) - 1:
                # 冻结 GraphPooling 内部 fc1, fc2 和 centroid_fc 的参数
                for param in pool.pool_block.fc1.parameters():
                    param.requires_grad = False
                for param in pool.pool_block.fc2.parameters():
                    param.requires_grad = False
                for param in pool.pool_block.centroid_fc.parameters():
                    param.requires_grad = False
            pools.append(pool)

        # 将块和池化层分配给类的成员变量，以便在 forward 中按阶段调用
        self.blocks1, self.pool1 = blocks[0], pools[0]
        self.blocks2, self.pool2 = blocks[1], pools[1]
        self.blocks3, self.pool3 = blocks[2], pools[2]
        self.blocks4, self.pool4 = blocks[3], pools[3]
        # --------------------------------------------------------------------------


    def _block_operations(self, x, cls_token, x_pad_mask,
                          nn_block, pool_block, norm_block):
        """
        定义每个分层阶段的操作：Attention Blocks 前传 -> Pooling -> 归一化 (可选)。
        
        Args:
            x (Tensor): 当前阶段的输入 token (非 cls token)。
            cls_token (Tensor): 当前阶段的 cls token。
            x_pad_mask (Tensor): 当前 token 的填充掩码。
            nn_block (ModuleList): 当前阶段的 Attention Blocks。
            pool_block (Pooling): 当前阶段的 Pooling 层。
            norm_block (nn.Module): 归一化层 (如 self.norm)，最后一个阶段需要。
        
        Returns:
            Tuple: 包含下一阶段的输入 (x, cls_token, 各种中间结果, 最终输出 out)。
        """
        # 将 cls_token 与输入 token x 拼接，形成 Attention Blocks 的完整输入
        cls_x = torch.cat([cls_token, x], dim=1)
        cls_x = nn_block(cls_x).type_as(x)  # 经过 Attention Blocks 前传
        cls_token, x = cls_x[:, :1, :], cls_x[:, 1:, :]  # 分离出更新后的 cls_token 和 x

        # 仅对 x 执行池化操作 (Graph Pooling)
        # pool_block 返回下一阶段的 cls_token, 池化 logit, 聚类中心 (centroid, 即下一阶段的 x), 
        # 下一阶段的填充掩码, 和池化索引
        cls_token, pool_logit, centroid, pool_pad_mask, pool_inds = (
            pool_block(cls_token, x, x_pad_mask)
        )

        # 基于 cls_token 生成阶段输出 (out)
        if norm_block is not None:
            # 最后一个阶段会使用 self.norm 归一化后取出 cls_token
            out = norm_block(cls_x)[:, 0]
        else:
            # 中间阶段直接取出 cls_token 作为阶段输出
            out = cls_x[:, 0]

        return (x, cls_token, pool_logit, centroid,
                pool_pad_mask, pool_inds, out)

    def forward_features(self, x, y): # x: B x 3 x 224 x 224, y: B x 224 x 224
        """
        前传特征提取部分。
        
        Args:
            x (Tensor): 输入图像 (B x 3 x H x W)，如 B x 3 x 224 x 224。
            y (Tensor): 语义分割/分块信息 (B x H x W)，如 B x 224 x 224。
            
        Returns:
            dict: 包含每个阶段中间结果的字典 (intermediates)。
        """
        x = self.patch_embed(x) # NxHxWxC Bx28x28x384
        N, H, W, C = x.shape
        # Collect features within each segment
        y = y.unsqueeze(1).float()
        y = F.interpolate(y, x.shape[1:3], mode='nearest')
        y = y.squeeze(1).long()  # Bx28x28
        x = segment_mean_nd(x, y) # Bx196x384   
        # Create padding mask
        ones = torch.ones((N, H, W, 1), dtype=x.dtype, device=x.device)
        avg_ones = segment_mean_nd(ones, y).squeeze(-1)
        x_padding_mask = avg_ones <= 0.5   ##S_0? 

        # Collect positional encodings within each segment
        pos_embed = self.pos_embed[:, 1:].view(1, H, W, C).expand(N, -1, -1, -1) #Bx28x28x384 (B만큼 복사됨?)
        pos_embed = segment_mean_nd(pos_embed, y)  #Bx196x384

        # Add positional encodings
        x = self.pos_drop(x + pos_embed)  # Bx196x384

        # Add class token.
        #self.cls_token: 1x1x384
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) #Bx1x384
        cls_token = cls_token + self.pos_embed[:, :1]

        # intermediate results
        intermediates = {}

        # Block1
        (block1, cls_token1, pool_logit1, centroid1,
         pool_padding_mask1, pool_inds1, out1) = self._block_operations(
            x, cls_token, x_padding_mask,
            self.blocks1, self.pool1, None)
        # cls_token1: Bx1x384, pool_padding_mask1: Bx64, centroid1: Bx64x384, out1: Bx384
        intermediates1 = {
            'logit1': pool_logit1, 'centroid1': centroid1, 'block1': block1,
            'padding_mask1': x_padding_mask, 'sampled_inds1': pool_inds1,
        }
        intermediates.update(intermediates1)

        # Block2
        (block2, cls_token2, pool_logit2, centroid2,
         pool_padding_mask2, pool_inds2, out2) = self._block_operations(
            centroid1, cls_token1, pool_padding_mask1,
            self.blocks2, self.pool2, None)
        # cls_token2: Bx1x384, pool_padding_mask2: Bx32, centroid2: Bx32x384, out2: Bx384
        intermediates2 = {
            'logit2': pool_logit2, 'centroid2': centroid2, 'block2': block2,
            'padding_mask2': pool_padding_mask1, 'sampled_inds2': pool_inds2, 'out2': out2, 
        }
        intermediates.update(intermediates2)

        # Block3
        (block3, cls_token3, pool_logit3, centroid3,
         pool_padding_mask3, pool_inds3, out3) = self._block_operations(
            centroid2, cls_token2, pool_padding_mask2,
            self.blocks3, self.pool3, None)
        # cls_token3: Bx1x384, pool_padding_mask3: Bx16, centroid3: Bx16x384, out3: Bx384
        intermediates3 = {
            'logit3': pool_logit3, 'centroid3': centroid3, 'block3': block3,
            'padding_mask3': pool_padding_mask2, 'sampled_inds3': pool_inds3, 'out3': out3,
        }
        intermediates.update(intermediates3)

        # Block4
        (block4, cls_token4, pool_logit4, centroid4,
         pool_padding_mask4, pool_inds4, out4) = self._block_operations(
            centroid3, cls_token3, pool_padding_mask3,
            self.blocks4, self.pool4, self.norm)
        # cls_token4: Bx1x384, pool_padding_mask4: Bx8, centroid4: Bx8x384, out4: Bx384
        out4 = self.pre_logits(out4)
        intermediates4 = {
            'logit4': pool_logit4, 'centroid4': centroid4, 'block4': block4,
            'padding_mask4': pool_padding_mask3, 'out4': out4, 'sampled_inds4': pool_inds4,
        }
        intermediates.update(intermediates4)

        return intermediates

    def forward(self, x, y):
        intermediates = self.forward_features(x, y)  # B x 384
        x = self.head(intermediates['out4']) # B x 1000

        return x


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
