# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
from rich import print

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
#from timm.models.vision_transformer import Attention, Mlp, RmsNorm, use_fused_attn
from timm.models.vision_transformer  import use_fused_attn,Mlp,RmsNorm

from models.taylor.cache_functions.attention import TaylorAttention 
from models.taylor.taylor_utils import derivative_approximation
from models.taylor.taylor_utils import taylor_cache_init, derivative_approximation,taylor_formula


from models.taylor.cache_functions import cache_init


import contextlib
@contextlib.contextmanager
def cuda_timer(name: str = None):
    results={}
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    yield   results               
    end.record()
    torch.cuda.synchronize()
    results['ms'] = start.elapsed_time(end)
    
    # if name:
    #     print(f"[red]{name}：{ms:.3f}ms[/red]")
    #     #print(f"[{name}] {ms:.3f} ms")
    


#################################################################################
#               Embedding Layers for Timesteps and Condition Inptus             #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.bfloat16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                          Cross Attention Layers                               #
#################################################################################
class CrossAttention(nn.Module):
    """
    A cross-attention layer with flash attention.
    """
    fused_attn: Final[bool]
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0,
            proj_drop: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, 
                mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, C = x.shape
        _, L, _ = c.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(c).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Prepare attn mask (B, L) to mask the conditioion
        if mask is not None:
            mask = mask.reshape(B, 1, 1, L)
            mask = mask.expand(-1, -1, N, -1)
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=mask
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill_(mask.logical_not(), float('-inf'))
            attn = attn.softmax(dim=-1)
            if self.attn_drop.p > 0:
                attn = self.attn_drop(attn)
            x = attn @ v
            
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        if self.proj_drop.p > 0:
            x = self.proj_drop(x)
        return x

class TaylorCrossAttention(nn.Module):
    """
    A cross-attention layer with flash attention.
    """
    fused_attn: Final[bool]
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0,
            proj_drop: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, 
                mask: torch.Tensor | None = None,cache_dic=None, current=None,) -> torch.Tensor:
        B, N, C = x.shape
        _, L, _ = c.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(c).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Prepare attn mask (B, L) to mask the conditioion
        if mask is not None:
            mask = mask.reshape(B, 1, 1, L)
            mask = mask.expand(-1, -1, N, -1)
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=mask
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill_(mask.logical_not(), float('-inf'))
            attn = attn.softmax(dim=-1)
            if self.attn_drop.p > 0:
                attn = self.attn_drop(attn)
            x = attn @ v
            
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        if self.proj_drop.p > 0:
            x = self.proj_drop(x)
        return x
#################################################################################
#                                 RDT Block                                     #
#################################################################################
class RDTBlock_taylor(nn.Module):
    """
    A RDT block with cross-attention conditioning.
    """
    def __init__(self, hidden_size, num_heads, time_records,**block_kwargs):
        super().__init__()
        self.time_records=time_records
        self.norm1 = RmsNorm(hidden_size, eps=1e-6)
        #self.hidden_size=hidden_size
        #GPT
        self.attn = TaylorAttention(
            dim=hidden_size, num_heads=num_heads, 
            qkv_bias=True, qk_norm=True, 
            norm_layer=RmsNorm,**block_kwargs)
        
        
        self.cross_attn = TaylorCrossAttention(
            hidden_size, num_heads=num_heads, 
            qkv_bias=True, qk_norm=True, 
            norm_layer=RmsNorm,**block_kwargs)
    
        
        self.norm2 = RmsNorm(hidden_size, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn = WrappedTaylorMlp(
            in_features=hidden_size,
            hidden_features=hidden_size,
            act_layer=approx_gelu, drop=0)
        #flag
        # self.ffn = Mlp(in_features=hidden_size, 
        #     hidden_features=hidden_size, 
        #     act_layer=approx_gelu, drop=0)
        self.norm3 = RmsNorm(hidden_size, eps=1e-6)
    #ori forward
    # def forward(self, x, c, mask=None):
    
    #     #---------------------
    #     origin_x = x
    #     x = self.norm1(x)
    #     x = self.attn(x)
    #     x = x + origin_x
        
    #     origin_x = x
    #     x = self.norm2(x)
    #     x = self.cross_attn(x, c, mask)
    #     x = x + origin_x
                
    #     origin_x = x
    #     x = self.norm3(x)
    #     x = self.ffn(x)
    #     x = x + origin_x
        
    #     return x

    def forward(self, x, c, mask=None, *, cache_dic, current):
        """
        x      : (B, T, D) 主序列 (state+action)
        c      : (B, L, D) cross-attention 条件
        mask   : (B, L)    bool mask
        cache_dic, current 见 ToCa 各函数
        """
        from colorama import init
        from termcolor import colored
        init()  # 在 Windows 上初始化支持
        # ------ Self-Attention with ToCa (module='attn') ------
        origin = x
        x_norm = self.norm1(x)
        
        current['module'] = 'attn'
        if current['type'] == 'full':
            if current['layer']==0:
                print(colored(f"full: ", "blue")+ colored(f"step:{current['step']:.3f}  layer:{current['layer']:.3f} ", "red"))
            with cuda_timer("full atten") as r:
                taylor_cache_init(cache_dic, current)
                sa_out = self.attn(x_norm, cache_dic, current) # ★ 把两个字典传进去
                derivative_approximation(cache_dic, current, sa_out)
            print(f"[red]full sa：{r['ms']:.3f}ms[/red]")
            self.time_records['full-sa'].append(r['ms'])
            # self.time_records = {
            # 'full-sa': [],        # 存每层 self-attn 的耗时
            # 'full-ca': [],  # 存每层 cross-attn 的耗时
            # 'full-mlp': [] ,         # 存每层 FFN 的耗时
            # 'taylor-sa': [],        # 存每层 self-attn 的耗时
            # 'taylor-ca': [],  # 存每层 cross-attn 的耗时
            # 'taylor-mlp': [] ,
            # 'flayer':[]
            # }
        else:  # Taylor 近似
            order = current.get('taylor_order', None)
            if current['layer']==0:
                print(colored(f"Taylor: ", "blue")+ colored(f"step:{current['step']:.3f}  layer:{current['layer']:.3f} ", "red"))
            with cuda_timer("full atten") as r:
                sa_out = taylor_formula(cache_dic, current, order_limit=order)
            print(f"[red]taylor sa：{r['ms']:.3f}ms[/red]")
            self.time_records['taylor-sa'].append(r['ms'])
        x = origin + sa_out  # 残差

        # ------ Cross-Attention（保持全量计算，不进缓存） ------
        origin = x
        q = self.norm2(x)
        current['module'] = 'cross_attn'
        if current['type'] == 'full':
            if current['layer'] == 0:
                print(colored(f"full (cross): ", "blue") +
                    colored(f"step:{current['step']:.3f}  layer:{current['layer']:.3f}", "red"))
            with cuda_timer("full cross_attn") as r:
                
                taylor_cache_init(cache_dic, current)
                ca_out = self.cross_attn(q, c, mask,cache_dic, current)
                #sa_out = self.attn(x_norm, cache_dic, current) # ★ 把两个字典传进去
                #bug  luo zhongwei
                #这里应该写成derivative_approximation(cache_dic, current, ca_out)
                #逆天的是居然发现写错，写成sa 还会高几个点，这个就非常神奇？？？？？
                derivative_approximation(cache_dic, current, ca_out)
                
            self.time_records['full-ca'].append(r['ms'])
            print(f"[red]full cross_attn：{r['ms']:.3f}ms[/red]")
        else:
            order = current.get('taylor_order', None)
            if current['layer'] == 0:
                print(colored(f" Taylor (cross): ", "blue") +
                    colored(f"step:{current['step']:.3f}  layer:{current['layer']:.3f}", "red"))
            with cuda_timer("taylor cross_attn") as r:
                ca_out = taylor_formula(cache_dic, current,order_limit=order)
            print(f"[red]taylor cross_attn：{r['ms']:.3f}ms[/red]")
            self.time_records['taylor-ca'].append(r['ms'])
        x = origin + ca_out

        # ------ FFN with ToCa (module='mlp') ------
        origin = x
        x_norm = self.norm3(x)
        current['module'] = 'mlp'
        if current['type'] == 'full':
            with cuda_timer("full atten") as r:
                ffn_out = self.ffn(x_norm, cache_dic, current)
            print(f"[red]full mlp：{r['ms']:.3f}ms[/red]")
            self.time_records['full-mlp'].append(r['ms'])
        else:
            order = current.get('taylor_order', None)
            with cuda_timer("full atten") as r:
                ffn_out = taylor_formula(cache_dic, current, order_limit=order)
            print(f"[red]taylor mlp：{r['ms']:.3f}ms[/red]")
            self.time_records['taylor-mlp'].append(r['ms'])
        x = origin + ffn_out
        return x
class FinalLayer(nn.Module):
    """
    The final layer of RDT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RmsNorm(hidden_size, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn_final = Mlp(in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_channels, 
            act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.ffn_final(x)
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    if not isinstance(pos, np.ndarray):
        pos = np.array(pos, dtype=np.float64)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_nd_sincos_pos_embed_from_grid(embed_dim, grid_sizes):
    """
    embed_dim: output dimension for each position
    grid_sizes: the grids sizes in each dimension (K,).
    out: (grid_sizes[0], ..., grid_sizes[K-1], D)
    """
    num_sizes = len(grid_sizes)
    # For grid size of 1, we do not need to add any positional embedding
    num_valid_sizes = len([x for x in grid_sizes if x > 1])
    emb = np.zeros(grid_sizes + (embed_dim,))
    # Uniformly divide the embedding dimension for each grid size
    dim_for_each_grid = embed_dim // num_valid_sizes
    # To make it even
    if dim_for_each_grid % 2 != 0:
        dim_for_each_grid -= 1
    valid_size_idx = 0
    for size_idx in range(num_sizes):
        grid_size = grid_sizes[size_idx]
        if grid_size <= 1:
            continue
        pos = np.arange(grid_size)
        posemb_shape = [1] * len(grid_sizes) + [dim_for_each_grid]
        posemb_shape[size_idx] = -1
        emb[..., valid_size_idx * dim_for_each_grid:(valid_size_idx + 1) * dim_for_each_grid] += \
            get_1d_sincos_pos_embed_from_grid(dim_for_each_grid, pos).reshape(posemb_shape)
        valid_size_idx += 1
    return emb


def get_multimodal_cond_pos_embed(embed_dim, mm_cond_lens: OrderedDict, 
                                  embed_modality=True):
    """
    Generate position embeddings for multimodal conditions. 
    
    mm_cond_lens: an OrderedDict containing 
        (modality name, modality token length) pairs.
        For `"image"` modality, the value can be a multi-dimensional tuple.
        If the length < 0, it means there is no position embedding for the modality or grid.
    embed_modality: whether to embed the modality information. Default is True.
    """
    num_modalities = len(mm_cond_lens)
    modality_pos_embed = np.zeros((num_modalities, embed_dim))
    if embed_modality:
        # Get embeddings for various modalites
        # We put it in the first half
        modality_sincos_embed = get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2, torch.arange(num_modalities))
        modality_pos_embed[:, :embed_dim // 2] = modality_sincos_embed
        # The second half is for position embeddings
        pos_embed_dim = embed_dim // 2
    else:
        # The whole embedding is for position embeddings
        pos_embed_dim = embed_dim
    
    # Get embeddings for positions inside each modality
    c_pos_emb = np.zeros((0, embed_dim))
    for idx, (modality, cond_len) in enumerate(mm_cond_lens.items()):
        if modality == "image" and \
            (isinstance(cond_len, tuple) or isinstance(cond_len, list)):
            all_grid_sizes = tuple([abs(x) for x in cond_len])
            embed_grid_sizes = tuple([x if x > 0 else 1 for x in cond_len])
            cond_sincos_embed = get_nd_sincos_pos_embed_from_grid(
                pos_embed_dim, embed_grid_sizes)
            cond_pos_embed = np.zeros(all_grid_sizes + (embed_dim,))
            cond_pos_embed[..., -pos_embed_dim:] += cond_sincos_embed
            cond_pos_embed = cond_pos_embed.reshape((-1, embed_dim))
        else:
            cond_sincos_embed = get_1d_sincos_pos_embed_from_grid(
                pos_embed_dim, torch.arange(cond_len if cond_len > 0 else 1))
            cond_pos_embed = np.zeros((abs(cond_len), embed_dim))
            cond_pos_embed[:, -pos_embed_dim:] += cond_sincos_embed
        cond_pos_embed += modality_pos_embed[idx]
        c_pos_emb = np.concatenate([c_pos_emb, cond_pos_embed], axis=0)
    
    return c_pos_emb



class WrappedTaylorAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.taylor_attn = TaylorAttention(*args, **kwargs)

    def forward(self, x, cache_dic, current):
        taylor_cache_init(cache_dic, current)
        out = self.taylor_attn(x, cache_dic=cache_dic, current=current)
        derivative_approximation(cache_dic, current, out)
        return out
    
# class WrappedTaylorMlp(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.mlp = Mlp(*args, **kwargs)

#     def forward(self, x, cache_dic, current):
#         # 第一次进入该模块时建缓存
#         taylor_cache_init(cache_dic, current)
#         out = self.mlp(x)
#         derivative_approximation(cache_dic, current, out)
#         return out
    
class WrappedTaylorMlp(nn.Module):
    """保持原有参数名 fc1 / fc2，便于加载预训练权重"""
    def __init__(self, in_features, hidden_features, act_layer, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        #self.drop = nn.Dropout(drop)

    def forward(self, x, cache_dic=None, current=None):
        if cache_dic is not None and current is not None:
            taylor_cache_init(cache_dic, current)
        x = self.fc1(x)
        x = self.act(x)
        #x = self.drop(x)
        x = self.fc2(x)
        if cache_dic is not None and current is not None:
            derivative_approximation(cache_dic, current, x)
        return x
    
# class Mlp(nn.Module):
#     """ MLP as used in Vision Transformer, MLP-Mixer and related networks

#     NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
#     """
#     def __init__(
#             self,
#             in_features,
#             hidden_features=None,
#             out_features=None,
#             act_layer=nn.GELU,
#             norm_layer=None,
#             bias=True,
#             drop=0.,
#             use_conv=False,
#     ):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         bias = to_2tuple(bias)
#         drop_probs = to_2tuple(drop)
#         linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

#         self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
#         self.act = act_layer()
#         self.drop1 = nn.Dropout(drop_probs[0])
#         self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
#         self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
#         self.drop2 = nn.Dropout(drop_probs[1])

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop1(x)
#         x = self.norm(x)
#         x = self.fc2(x)
#         x = self.drop2(x)
#         return x