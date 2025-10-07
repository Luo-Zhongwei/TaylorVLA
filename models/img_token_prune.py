# utils/prune_tokens.py (GPU-Optimized Version)

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 使用无窗口后端
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw
from typing import List, Sequence
from pathlib import Path
from datetime import datetime

# --- GPU原生计算函数 (无需修改，已是GPU友好) ---

@torch.no_grad()
def compute_attn_score_with_rdt_head(text_cond, img_cond, cross_attn_module):
    """
    计算Cross-Attention分数，全程在GPU上运行。
    返回: attn_score -> (B, L_img) 的张量。
    """
    B, L_txt, D = text_cond.shape
    _, L_img, _ = img_cond.shape
    H = cross_attn_module.num_heads
    head_dim = D // H

    q = cross_attn_module.q(text_cond)
    kv = cross_attn_module.kv(img_cond)
    k, _ = kv.chunk(2, dim=-1) # 只需要K

    q = q.view(B, L_txt, H, head_dim).permute(0, 2, 1, 3)
    k = k.view(B, L_img, H, head_dim).permute(0, 2, 1, 3)

    scale = head_dim ** -0.5
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = attn_logits.softmax(dim=-1)

    attn_score = attn.mean(dim=(1, 2)) # 在Head和Text Token维度上求平均
    return attn_score

@torch.no_grad()
def prune_tokens_random(tokens: torch.Tensor, topk: int):
    """
    随机剪枝，全程在GPU上运行。
    """
    B, L, D = tokens.shape
    rand_indices = torch.rand(B, L, device=tokens.device).topk(k=topk, dim=-1).indices
    keep_idx_sorted, _ = torch.sort(rand_indices, dim=-1)
    
    batch_idx = torch.arange(B, device=tokens.device)[:, None]
    pruned = tokens[batch_idx, keep_idx_sorted, :]
    return pruned, keep_idx_sorted


# --- 【核心修改】评分剪枝函数 ---

@torch.no_grad()
def prune_tokens_with_scores(
    tokens: torch.Tensor, 
    attn_score: torch.Tensor, 
    alpha: float = 0.7, 
    sim_temp: float = 0.07, 
    topk: int = 256
):
    """
    【GPU优化版】根据注意力分数和相似度冗余进行剪枝。
    所有计算都在GPU上完成。
    
    Args:
        tokens (torch.Tensor): (B, L, D)
        attn_score (torch.Tensor): (B, L)
    Returns:
        pruned_tokens (torch.Tensor), keep_idx_sorted (torch.Tensor)
    """
    B, L, D = tokens.shape

    # 1. 计算冗余度 (在GPU上)
    tokens_n = F.normalize(tokens, dim=-1)
    sim = torch.matmul(tokens_n, tokens_n.transpose(1, 2))
    
    # 填充对角线以排除自身与自身的比较
    sim.diagonal(dim1=-2, dim2=-1).fill_(-float('inf'))
    # 在对角线填充-inf之前，克隆一份原始的sim矩阵用于返回
    sim_for_return = sim.clone()
    
    
    
    redundancy = sim.max(dim=-1).values
    redundancy = torch.sigmoid(redundancy / sim_temp) # (B, L)

    # 2. 归一化注意力分数 (在GPU上)
    # 逐个样本归一化 (dim=-1)
    min_val = torch.min(attn_score, dim=-1, keepdim=True)[0]
    max_val = torch.max(attn_score, dim=-1, keepdim=True)[0]
    attn_score_normalized = (attn_score - min_val) / (max_val - min_val + 1e-6)

    # 3. 计算最终分数 (在GPU上)
    # alpha 越大，越关注注意力；(1-alpha) 越大，越关注去冗余
    final_score = alpha * attn_score_normalized + (1 - alpha) * (1 - redundancy)

    # 4. 选取top-k的索引 (在GPU上)
    keep_idx = torch.topk(final_score, k=topk, dim=-1).indices
    keep_idx_sorted, _ = torch.sort(keep_idx, dim=-1)

    # 5. 根据索引提取tokens (在GPU上)
    batch_idx = torch.arange(B, device=tokens.device)[:, None]
    pruned_tokens = tokens[batch_idx, keep_idx_sorted, :]

    return pruned_tokens, keep_idx_sorted,sim_for_return


# --- 【核心修改】可视化函数 ---
# 这些函数仍然在CPU上运行，但在被调用时才从GPU获取数据

def overlay_grid(img: Image.Image, R: int, C: int) -> Image.Image:
    # (此函数无需修改，它操作的是PIL Image)
    overlay = img.convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    for r in range(1, R):
        draw.line([(0, r * h / R), (w, r * h / R)], fill=(255, 255, 255, 180))
    for c in range(1, C):
        draw.line([(c * w / C, 0), (c * w / C, h)], fill=(255, 255, 255, 180))
    return overlay

def overlay_prune_mask(img: Image.Image, R: int, C: int, keep_idx: np.ndarray, alpha=200, color=(128, 128, 128)) -> Image.Image:
    # (此函数无需修改，它操作的是PIL Image和NumPy array)
    overlay = img.convert("RGBA")
    mask = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)
    w, h = img.size
    cw, ch = w / C, h / R
    total = R * C
    pruned = set(range(total)) - set(keep_idx.tolist())
    for idx in pruned:
        r, c = divmod(idx, C)
        x0, y0 = c * cw, r * ch
        fill = (*color, alpha)
        draw.rectangle([x0, y0, x0 + cw, y0 + ch], fill=fill)
    comp = Image.alpha_composite(overlay, mask)
    return overlay_grid(comp, R, C)

def debug_visualize(
    images: List[Image.Image],
    grid_rows: int,
    grid_cols: int,
    keep_idxs: List[torch.Tensor],      # <-- 接收Tensor
    attn_grids: List[torch.Tensor],     # <-- 接收Tensor
    sim_matrices: List[torch.Tensor],   # <-- 接收Tensor
    save_vis_doc_name: str
):
    """
    【GPU优化版】接收Tensor，在函数内部进行必要的CPU转换和Reshape。
    """
    out_dir = Path("visual_debug") / Path(save_vis_doc_name) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(images)
    fig_all = plt.figure(figsize=(12, 4 * n))

    for i in range(n):
        img = images[i]
        
        # --- 在这里进行数据转换 ---
        keep_idx_np = keep_idxs[i].cpu().numpy()
        # 将一维注意力分数reshape成二维并转到CPU
        attn_grid_np = attn_grids[i].reshape(grid_rows, grid_cols).cpu().numpy()
        sim_matrix_np = sim_matrices[i].cpu().numpy()
        
        # ... 后续绘图代码不变 ...
        ax0 = fig_all.add_subplot(n, 4, 4 * i + 1)
        ax0.imshow(overlay_grid(img, grid_rows, grid_cols))
        ax0.set_title(f"View {i+1} ori")
        ax0.axis("off")

        ax1 = fig_all.add_subplot(n, 4, 4 * i + 2)
        ax1.imshow(overlay_prune_mask(img, grid_rows, grid_cols, keep_idx_np, color=(0,0,0), alpha=180))
        ax1.set_title(f"View {i+1} token p")
        ax1.axis("off")

        ax2 = fig_all.add_subplot(n, 4, 4 * i + 3)
        sns.heatmap(attn_grid_np, ax=ax2, cmap="viridis", cbar=False)
        ax2.set_title("Cross-Attn")
        ax2.axis("off")

        ax3 = fig_all.add_subplot(n, 4, 4 * i + 4)
        sns.heatmap(sim_matrix_np, ax=ax3, cmap="coolwarm", vmin=-1, vmax=1) # vmin可以调整
        ax3.set_title("Self-Sim")
        ax3.axis("off")

    plt.tight_layout()
    out_all = out_dir / "all_views.png"
    fig_all.savefig(out_all, dpi=100, bbox_inches="tight")
    plt.close(fig_all)

# --- 工具函数 (无需修改) ---

def ensure_pil_list(images: Sequence) -> List[Image.Image]:
    # ... (此函数无需修改)
    to_pil = transforms.ToPILImage()
    pil_list: List[Image.Image] = []
    if isinstance(images, (torch.Tensor, np.ndarray, Image.Image)):
        images = [images]
    for img in images:
        if isinstance(img, Image.Image):
            pil_list.append(img)
        elif isinstance(img, torch.Tensor):
            pil_list.append(to_pil(img.cpu()))
        elif isinstance(img, np.ndarray):
            pil_list.append(Image.fromarray(img.astype(np.uint8)))
        else:
            raise TypeError(f"Unsupported type: {type(img)}")
    return pil_list

def tensor_list_to_pil(tensor_list: List[torch.Tensor], image_mean: List[float], image_std: List[float]) -> List[Image.Image]:
    # ... (此函数无需修改)
    pil_images: List[Image.Image] = []
    mean_t = torch.tensor(image_mean, device=tensor_list[0].device)[:, None, None]
    std_t = torch.tensor(image_std, device=tensor_list[0].device)[:, None, None]
    for t in tensor_list:
        img = t * std_t + mean_t
        img = img.clamp(0, 1)
        img = (img * 255).to(torch.uint8)
        arr = img.permute(1, 2, 0).cpu().numpy()
        pil_images.append(Image.fromarray(arr))
    return pil_images