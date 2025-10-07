import torch.nn.functional as F
import torch
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

# @torch.no_grad()
# def prune_tokens_with_cdp(
#     tokens: torch.Tensor, 
#     lang_cond: torch.Tensor,
#     topk: int
# ):
#     """
#     使用 Conditional DPP (CDPruner) 逻辑对单个视图的图像令牌进行剪枝。
    
#     Args:
#         tokens (torch.Tensor): 单个视图的图像令牌 (B, L, D), 这里 B 通常为 1。
#         lang_cond (torch.Tensor): 语言条件令牌 (B, L_txt, D)。
#         topk (int): 需要保留的令牌数量。
        
#     Returns:
#         pruned_tokens (torch.Tensor): 剪枝后的图像令牌 (B, topk, D)。
#         keep_idx_sorted (torch.Tensor): 保留的令牌索引，已排序 (B, topk)。
#     """
#     B, N, D = tokens.shape
#     device = tokens.device

#     # Step 1: 计算相似度 (Diversity)
#     # 注意：这里的tokens已经是经过adaptor投影后的img_cond，可以直接使用
#     tokens_normalized = F.normalize(tokens, p=2, dim=-1).float()
#     similarity = torch.matmul(tokens_normalized, tokens_normalized.transpose(1, 2)) # (B, N, N)

#     # Step 2: 计算相关性 (Relevance)
#     # CDPruner原文使用未投影的CLIP特征计算相关性，这里我们做一个适配，
#     # 使用投影后的特征与语言特征计算某种形式的“相关性”。
#     # 为了简化并复用现有输入，我们用投影后的图像和语言特征的cosine similarity来近似。
#     lang_cond_normalized = F.normalize(lang_cond, p=2, dim=-1).float()
#     # (B, N, D) @ (B, D, L_txt) -> (B, N, L_txt)
#     relevance_matrix = torch.matmul(tokens_normalized, lang_cond_normalized.transpose(1, 2))
#     # 对所有语言token取最大值或平均值作为每个图像token的相关性分数
#     relevance = relevance_matrix.mean(dim=-1) # (B, N)
#     # 归一化
#     relevance = (relevance - relevance.min(dim=-1, keepdim=True).values + 1e-6) / \
#                 (relevance.max(dim=-1, keepdim=True).values - relevance.min(dim=-1, keepdim=True).values)

#     # Step 3: 构建核矩阵 (Kernel Matrix)
#     kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1) # (B, N, N)

#     # Step 4: 快速 MAP 推断 (Greedy Selection)
#     cis = torch.zeros((topk, B, N), device=device) # (T, B, N)
#     di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone() # (B, N)
#     select_idx = torch.empty((topk, B), dtype=torch.long, device=device) # (T, B)
    
#     for i in range(topk):
#         j = torch.argmax(di2s, dim=-1) # (B,)
#         select_idx[i] = j

#         # Cholesky分解的向量化版本
#         # eis = (kernel[:, j] - torch.einsum('tb,tbn->bn', cis[:i, torch.arange(B), j], cis[:i])) / torch.sqrt(di2s[:, j]).unsqueeze(-1)
#         # 上面的写法在B>1时有问题，因为j是(B,)-tensor。需要逐个batch索引
#         kernel_selected = kernel[torch.arange(B), j] # (B, N)
#         if i > 0:
#             # cis[:i] -> (i, B, N)
#             # cis[:i, torch.arange(B), j] -> (i, B)
#             # torch.einsum('ib,ibn->bn', cis[:i, torch.arange(B), j], cis[:i]) -> (B,N) is not supported
#             # Let's do it with a loop for batch, as B is usually 1
#             einsum_term = torch.zeros_like(kernel_selected)
#             for b_idx in range(B):
#                 einsum_term[b_idx] = torch.einsum('i,in->n', cis[:i, b_idx, j[b_idx]], cis[:i, b_idx, :])
            
#             eis = (kernel_selected - einsum_term) / torch.sqrt(di2s[torch.arange(B), j] + 1e-8).unsqueeze(-1)
#         else:
#             eis = kernel_selected / torch.sqrt(di2s[torch.arange(B), j] + 1e-8).unsqueeze(-1)

#         cis[i] = eis
#         di2s -= torch.square(eis)
#         di2s[torch.arange(B), j] = -float('inf')

#     # 整理并返回结果
#     select_idx_transposed = select_idx.t() # (B, T)
#     keep_idx_sorted, _ = torch.sort(select_idx_transposed, dim=-1)
    
#     batch_idx = torch.arange(B, device=tokens.device)[:, None]
#     pruned_tokens = tokens[batch_idx, keep_idx_sorted, :]
    
#     return pruned_tokens, keep_idx_sorted



@torch.no_grad()
def prune_tokens_with_cdp(
    tokens: torch.Tensor, 
    lang_cond: torch.Tensor,
    topk: int,
    theta: float = 0.5  # <--- 新增 theta 超参数，并设置默认值为 0.5
):
    """
    使用 Conditional DPP (CDPruner) 逻辑对单个视图的图像令牌进行剪枝。
    
    Args:
        tokens (torch.Tensor): 单个视图的图像令牌 (B, L, D)。
        lang_cond (torch.Tensor): 语言条件令牌 (B, L_txt, D)。
        topk (int): 需要保留的令牌数量。
        theta (float): 控制相关性和多样性平衡的超参数, 范围 [0, 1)。
                       0 表示纯多样性，接近 1 表示纯相关性。
    """
    B, N, D = tokens.shape
    device = tokens.device

    # Step 1: 计算相似度 (Diversity)
    tokens_normalized = F.normalize(tokens, p=2, dim=-1).float()
    similarity = torch.matmul(tokens_normalized, tokens_normalized.transpose(1, 2))

    # Step 2: 计算原始相关性 (Relevance)
    lang_cond_normalized = F.normalize(lang_cond, p=2, dim=-1).float()
    relevance_matrix = torch.matmul(tokens_normalized, lang_cond_normalized.transpose(1, 2))
    relevance = relevance_matrix.mean(dim=-1)
    relevance = (relevance - relevance.min(dim=-1, keepdim=True).values + 1e-6) / \
                (relevance.max(dim=-1, keepdim=True).values - relevance.min(dim=-1, keepdim=True).values + 1e-6)

    # --- 新增步骤：使用 theta 调整相关性分数 ---
    if theta > 0:
        # 防止 theta=1 导致分母为0
        if theta >= 1.0: theta = 0.999 
        alpha = theta / (2 * (1 - theta))
        relevance = torch.exp(alpha * relevance)
    # -----------------------------------------

    # Step 3: 构建核矩阵 (Kernel Matrix)
    kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)

    # ... (Step 4: 快速 MAP 推断部分保持不变)
    cis = torch.zeros((topk, B, N), device=device)
    di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone()
    select_idx = torch.empty((topk, B), dtype=torch.long, device=device)
    
    for i in range(topk):
        j = torch.argmax(di2s, dim=-1)
        select_idx[i] = j
        kernel_selected = kernel[torch.arange(B), j]
        if i > 0:
            einsum_term = torch.zeros_like(kernel_selected)
            for b_idx in range(B):
                einsum_term[b_idx] = torch.einsum('i,in->n', cis[:i, b_idx, j[b_idx]], cis[:i, b_idx, :])
            eis = (kernel_selected - einsum_term) / torch.sqrt(di2s[torch.arange(B), j] + 1e-8).unsqueeze(-1)
        else:
            eis = kernel_selected / torch.sqrt(di2s[torch.arange(B), j] + 1e-8).unsqueeze(-1)
        cis[i] = eis
        di2s -= torch.square(eis)
        di2s[torch.arange(B), j] = -float('inf')

    select_idx_transposed = select_idx.t()
    keep_idx_sorted, _ = torch.sort(select_idx_transposed, dim=-1)
    batch_idx = torch.arange(B, device=tokens.device)[:, None]
    pruned_tokens = tokens[batch_idx, keep_idx_sorted, :]
    
    return pruned_tokens, keep_idx_sorted





@torch.no_grad()
def prune_tokens_with_cdp_V2(
    tokens: torch.Tensor, 
    lang_cond: torch.Tensor,
    topk: int,
    theta: float = 0.5
):
    """
    【修正版】使用 Conditional DPP (CDPruner) 逻辑对单个视图的图像令牌进行剪枝。
    此版本更忠于原始论文和官方实现。

    Args:
        tokens (torch.Tensor): 单个视图的图像令牌 (B, L, D)，通常 B=1。
                                推荐使用原始CLIP图像特征。
        lang_cond (torch.Tensor): 语言条件令牌 (B, L_txt, D)，通常 B=1。
                                  推荐使用原始CLIP文本特征。
        topk (int): 需要保留的令牌数量。
        theta (float): 控制相关性和多样性平衡的超参数, 范围 [0, 1)。
                       0 表示纯多样性，接近 1 表示纯相关性。
                       
    Returns:
        pruned_tokens (torch.Tensor): 剪枝后的图像令牌 (B, topk, D)。
        keep_idx_sorted (torch.Tensor): 保留的令牌索引，已排序 (B, topk)。
    """
    # 0. 初始化与维度检查
    # 该实现主要针对 B=1 的场景，这在您的逐视图处理中是常见情况。
    B, N, D = tokens.shape
    if B > 1:
        # 如果需要处理批次，可以逐个样本循环调用此函数
        raise NotImplementedError("此版CDPruner实现为 B=1 优化，更简洁高效。")
    
    device = tokens.device

    # Step 1: 计算相似度核 (Diversity Kernel S)
    # 代表了token之间的“差异性”。两个token越相似，Sij值越大。
    # 我们希望选出的token集合彼此间的Sij值较小。
    tokens_normalized = F.normalize(tokens.float(), p=2, dim=-1)
    similarity = torch.matmul(tokens_normalized, tokens_normalized.transpose(1, 2)).squeeze(0) # (N, N)

    # Step 2: 计算相关性分数 (Relevance Score q)
    # 代表了每个图像token与文本指令的“相关性”。
    #lang_cond_normalized = F.normalize(lang_cond, p=2, dim=-1, dtype=torch.float32)
    lang_cond_normalized = F.normalize(lang_cond.float(), p=2, dim=-1)
    # (1, N, D) @ (1, D, L_txt) -> (1, N, L_txt)
    relevance_matrix = torch.matmul(tokens_normalized, lang_cond_normalized.transpose(1, 2)).squeeze(0) # (N, L_txt)
    
    # 核心修正：使用 max() 而非 mean()，与论文保持一致
    relevance = torch.max(relevance_matrix, dim=1)[0] # (N,)
    
    # 归一化到 [0, 1] 区间
    min_val, max_val = relevance.min(), relevance.max()
    relevance = (relevance - min_val + 1e-6) / (max_val - min_val + 1e-6)

    # Step 3: 应用 Theta 平衡因子
    # theta是核心超参数，用于平衡相关性(quality)和多样性(diversity)
    if theta > 0:
        if theta >= 1.0: theta = 0.999 # 防止分母为0
        alpha = theta / (2 * (1 - theta))
        relevance = torch.exp(alpha * relevance) # (N,)
    # 如果 theta=0, alpha=0, exp(0)=1, 所有token相关性都为1，只考虑多样性

    # Step 4: 构建最终的条件核矩阵 L
    # L_ij = relevance_i * similarity_ij * relevance_j
    # 一个元素对(i, j)在L中得分高，意味着i和j本身都很重要，并且它们彼此相似
    kernel = relevance.unsqueeze(1) * similarity * relevance.unsqueeze(0) # (N, N)

    # Step 5: 快速 MAP 推断 (Greedy Selection)
    # 这是DPP的核心，通过贪心算法高效地找出最能代表整体的topk个元素
    
    # 初始化
    cis = torch.zeros((topk, N), device=device)
    di2s = torch.diagonal(kernel).clone() # (N,) 每个元素自身的初始价值
    selected_indices = torch.empty(topk, dtype=torch.long, device=device)

    for i in range(topk):
        # 5a. 选择当前边际增益最大的元素
        j = torch.argmax(di2s)
        selected_indices[i] = j

        # 5b. Cholesky分解的向量化更新步骤
        # 计算新选中的元素j与所有其他元素的关系，同时考虑已选集合的影响
        if i > 0:
            # cis[:i, j] @ cis[:i, :] 计算了新元素j通过已选元素对其他元素产生的间接影响
            einsum_term = torch.einsum('i,in->n', cis[:i, j], cis[:i, :])
        else:
            einsum_term = 0
        
        # eis代表了元素j对其他元素的“新”信息量
        eis = (kernel[j, :] - einsum_term) / (torch.sqrt(di2s[j] + 1e-8))
        cis[i] = eis
        
        # 5c. 更新所有元素的边际增益
        # 从它们的价值中减去与新选中元素j相似的部分，实现了多样性惩罚
        di2s -= torch.square(eis)
        di2s[j] = -float('inf') # 确保不会再选中它

    # Step 6: 整理并返回结果
    keep_idx_sorted, _ = torch.sort(selected_indices, dim=0)
    keep_idx_sorted = keep_idx_sorted.unsqueeze(0) # 恢复batch维度 (1, topk)
    
    pruned_tokens = tokens[:, keep_idx_sorted.squeeze(0), :]
    
    return pruned_tokens, keep_idx_sorted


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


def visualize_cdp_pruning(
    images: List[Image.Image],
    grid_rows: int,
    grid_cols: int,
    keep_idxs_per_view: List[torch.Tensor],
    save_vis_doc_name: str
):
    """
    专为CDPruner设计的可视化函数.
    生成一个 6x2 的拼接图，展示剪枝前后的对比。
    
    参数:
        images (List[Image.Image]): 6个原始视图的PIL图像列表.
        grid_rows (int): Token网格的行数.
        grid_cols (int): Token网格的列数.
        keep_idxs_per_view (List[torch.Tensor]): 包含6个Tensor的列表,
            每个Tensor是一维的、对应视图保留的token的局部索引 (local indices).
        save_vis_doc_name (str): 用于创建保存目录的名称.
    """
    # 1. 创建保存目录，使用时间戳确保每次运行都生成新文件
    out_dir = Path("visual_debug_cdp") / Path(save_vis_doc_name) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    num_views = len(images)
    
    # 2. 创建一个 6x2 的子图网格
    fig, axes = plt.subplots(num_views, 2, figsize=(6, 3 * num_views))
    
    fig.suptitle('CDPruner Token 可视化', fontsize=16)

    for i in range(num_views):
        img = images[i]
        
        # 3. 将当前视图的 'keep_idxs' Tensor 转换到 CPU NumPy 数组
        #    这是调用后续绘图函数所必需的
        keep_idx_np = keep_idxs_per_view[i].cpu().numpy().flatten()

        # --- 左侧子图: 原始图像 + 网格 ---
        ax_left = axes[i, 0]
        img_with_grid = overlay_grid(img, grid_rows, grid_cols)
        ax_left.imshow(img_with_grid)
        ax_left.set_title(f"视图 {i+1} 原始图像")
        ax_left.axis("off")

        # --- 右侧子图: 剪枝后的图像 ---
        ax_right = axes[i, 1]
        # 使用 overlay_prune_mask 将被剪掉的patch用半透明蒙版覆盖
        img_pruned = overlay_prune_mask(img, grid_rows, grid_cols, keep_idx_np, color=(0, 0, 0), alpha=180)
        ax_right.imshow(img_pruned)
        ax_right.set_title(f"视图 {i+1} 剪枝后 ({len(keep_idx_np)} tokens)")
        ax_right.axis("off")

    # 4. 调整布局、保存并关闭图形，防止因循环调用导致内存泄漏
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    save_path = out_dir / "cdp_pruning_comparison.png"
    fig.savefig(save_path, dpi=120)
    plt.close(fig)