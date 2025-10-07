import torch
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

# 1. 设备与参数
device = "cuda:0"
dtype = torch.float16
batch, seqlen, embed = 4, 256, 128
num_heads = 8
head_dim = embed // num_heads

# 2a. 统一长度版本：分别生成 Q, K, V，形状 (batch, seqlen, nheads, head_dim)
q = torch.randn((batch, seqlen, num_heads, head_dim), device=device, dtype=dtype)
k = torch.randn_like(q)
v = torch.randn_like(q)

# 3a. 调用 flash_attn_func
out1 = flash_attn_func(q, k, v,
    dropout_p=0.0, softmax_scale=None, causal=False)
# 返回 (batch, seqlen, nheads, head_dim)
print("flash_attn_func output shape:", out1.shape)

# 2b. 打包 QKV 版本：形状 (batch, seqlen, 3, nheads, head_dim)
qkv = torch.stack([q, k, v], dim=2)

# 3b. 调用 flash_attn_qkvpacked_func
out2 = flash_attn_qkvpacked_func(qkv,
    dropout_p=0.0, softmax_scale=None, causal=False)
print("flash_attn_qkvpacked_func output shape:", out2.shape)