"""Define Graph Pooling."""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.geometry as dgl_geo
from cast_models.utils import segment_mean_nd


class Attention(nn.Module):
    """Similar to timm.models.vision_transformer.Attention but we do not use
    additional Fully Connected Layers.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Same as timm.models.vision_transformer.Block
    """

    def __init__(self, dim, num_heads, qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.bias = nn.Parameter(torch.zeros(dim).normal_(0, 1e-2))

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))

        x = x - torch.mean(x, dim=1, keepdim=True) + self.bias.view(1, 1, -1)

        return x


class GraphPooling(nn.Module):

    def __init__(self,
                 num_clusters=4,
                 d_model=512,
                 dropout=0.1,
                 l2_normalize_for_fps=True,
                 num_heads=12,
                 qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        """Perfrom Graph Pooling.
        执行图池化（Graph Pooling）以降采样特征序列。
        该模块结合了 Farthest Point Sampling (FPS) 和 Attention 机制进行软聚类。
        
        Args:
          num_clusters: A scalar indicates the number of centroids
          d_model: A scalar indicates the input channels to Transformer
          dropout: A `float` indicates the dropout rate
          l2_normalize_for_fps: Enable/disable L2-noramlization before performing
                                Farthest Point Sampling
          num_heads: A scalar indicates the number of attention head
          qkv_bias: Enable/disable bias in the attention layer
          norm_layer: A Norm layer used in the attention layer
        """
        super().__init__()
        # Centroid Feature Transformation Block：用于在进行聚类分配前，对输入特征进行一次自注意力转换。
        # 它可以帮助 Centroid/Query 更好地捕捉局部信息。
        self.centroid_fc = Block(dim=d_model,
                                 num_heads=num_heads,
                                 qkv_bias=qkv_bias,
                                 norm_layer=norm_layer)
        # 多层感知机 (MLP) 的第一层，用于在聚类/平均池化之前对特征进行变换。
        self.fc1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4, bias=True),
            nn.GELU(),
            nn.Dropout(dropout))
        # MLP 的第二层，用于在聚类/平均池化之后对特征进行变换。
        self.fc2 = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_model, bias=True))
        
        # 目标聚类数（即下一阶段的序列长度）
        self._num_clusters = num_clusters
        # 在执行 FPS 之前是否对特征向量进行 L2 归一化。
        self._l2_normalize_for_fps = l2_normalize_for_fps

    def _fill_with_mean(self, src, mask):
        """一个辅助函数，用于将输入序列中被掩码 (padded) 的无效条目填充为有效特征的均值。
        这有助于在后续的 FPS 采样中避免采样到无用的填充值。
        """
        """A helper function to fill invalid entries with mean values.
        """
        bs, sl, cs = src.shape
        if mask is not None:
            # 计算有效特征的均值 (valid_mean 假设是一个处理掩码的函数)
            mean_src = valid_mean(src, ~mask).unsqueeze(1).type_as(src)
            # 创建一个用于填充的扩展掩码 [B, SL, C]
            fill_mask = mask.unsqueeze(2).expand(-1, -1, cs)
            # 使用 torch.where 将填充位置替换为均值
            filled_src = torch.where(fill_mask, mean_src.expand(-1, sl, -1), src)
        else:
            # 如果没有掩码，直接计算序列均值
            mean_src = torch.mean(src, dim=1, keepdim=True).type_as(src)
            filled_src = src

        return filled_src, mean_src

    def forward(self, cls_token, src, mask):
        """使用 Transformer 进行聚类的前向传播。

        Args:
          cls_token: 形状 `[batch_size, 1, channels]` 的 cls token。
          src: 形状 `[batch_size, source_sequence_length, channels]` 的输入序列 token (x)。
          mask: 形状 `[batch_size, sequence_length]` 的布尔掩码，`True` 表示填充/无效元素。

        Returns:
          cls_token: 更新后的 cls token。
          centroids: 聚类中心 (下一阶段的 x)，形状 `[batch_size, num_clusters, channels]`。
          logits: 原始 token 到聚类中心的相似度/分配分数，形状 `[batch_size, SL, num_clusters]`。
          sampled_inds: 选定的聚类中心（Centroids）在原始序列中的索引。
        """
        bs, sl, cs = src.shape

        # --- 1. 准备和执行 Farthest Point Sampling (FPS) ---
        # 填充被掩码的特征，并获取有效特征的均值。
        filled_src, mean_src = self._fill_with_mean(src, mask)
        padded_src = torch.cat([mean_src, filled_src], dim=1)

        if self._l2_normalize_for_fps:
            sampling_src = F.normalize(padded_src, dim=-1)
        else:
            sampling_src = padded_src

        sampled_inds = dgl_geo.farthest_point_sampler(
            sampling_src.to(torch.float64),
            self._num_clusters + 1,
            0).long()
        sampled_inds = sampled_inds[:, 1:] - 1
        # assert((sampled_inds  >= 0).all()) # Make sure sampling from the squence
        if not (sampled_inds >= 0).all():
            sampled_inds = torch.clamp(sampled_inds, min=0)
        unfold_sampled_inds = sampled_inds.unsqueeze(2).expand(-1, -1, cs)

        # Apply attention layer to predict grouping
        node_features = self.centroid_fc(src)
        centroid_features = torch.gather(node_features, 1, unfold_sampled_inds)

        # Group squence of tokens into clusters
        normed_centroid_features = F.normalize(centroid_features, dim=-1)
        normed_node_features = F.normalize(node_features, dim=-1)
        logits = torch.einsum(
            'bij,bjk->bik', normed_node_features, normed_centroid_features.transpose(1, 2))
        logits = logits * 5
        assignments = torch.softmax(logits, dim=-1)

        # Average pooling within clusters.
        fc1_cls_token_src = self.fc1(torch.cat([cls_token, src], dim=1))
        fc1_cls_token, fc1_src = fc1_cls_token_src[:, :1], fc1_cls_token_src[:, 1:]

        # batch matrix multiplication
        normalizer = torch.einsum('bij,bjk->bik', assignments.transpose(1, 2),
                                  torch.ones((bs, sl, 1), dtype=src.dtype, device=src.device))
        centroids = torch.einsum('bij,bjk->bik', assignments.transpose(1, 2), fc1_src)
        centroids /= normalizer

        fc2_cls_token_centroids = self.fc2(torch.cat([fc1_cls_token, centroids], dim=1))
        centroids = fc2_cls_token_centroids[:, 1:, :] + torch.gather(src, 1, unfold_sampled_inds)
        cls_token = fc2_cls_token_centroids[:, :1, :] + cls_token

        return cls_token, centroids, logits, sampled_inds # centroids = Z, logits = 
        # logits 대신 assignments out 해야하는거 아니야? 


def valid_mean(x, mask):
     """Compute mean of x given valid mask.

     Args:
         x: A `float` tensor of shape `[batch_size, num_nodes, channels]`
         mask: A `bool` tensor of shape `[batch_size, num_nodes]`, where
             `True` indicates the entry is valid/padded

     Returns:
         mean_x: A `float` tensor of shape `[batch_size, channels]`
     """
     mask = mask.type_as(x).unsqueeze(2)
     sum_mask = torch.clamp(torch.sum(mask, dim=1), min=1)
     masked_x = x * mask
     mean_x = torch.sum(masked_x, dim=1) / sum_mask

     return mean_x

