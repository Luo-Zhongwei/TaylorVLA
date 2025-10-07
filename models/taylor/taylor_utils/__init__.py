from typing import Dict
import torch
import math

# def derivative_approximation(cache_dic, current, feature):
#     i = current['step_idx']
#     if i == 0:
#         diff = 1.0                     # 首步占位
#     else:
#         prev_t = current['activated_steps'][i-1]
#         curr_t = current['activated_steps'][i]
#         diff = prev_t - curr_t         # 200, 200, 199, …

#     updated = {0: feature}
#     for k in range(cache_dic['max_order']):
#         prev = cache_dic['cache'][-1][current['layer']][current['module']].get(k)
#         if prev is None:
#             break
#         updated[k+1] = (updated[k] - prev) / diff
#     cache_dic['cache'][-1][current['layer']][current['module']] = updated
# def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
#     """
#     Compute derivative approximation.
#     :param cache_dic: Cache dictionary.
#     :param current: Current step information.
#     """
#     hist = current['activated_steps']
#     if len(hist) < 2:
#         # 第一次进入还没有“上一次”可用，
#         # 直接退出或把导数置 0 都行
#         cache_dic['cache'][-1][current['layer']][current['module']] = {0: feature}
#         return
    
#     difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]
#     # difference_distance = current['activated_times'][-1] - current['activated_times'][-2]

#     updated_taylor_factors = {}
#     updated_taylor_factors[0] = feature
#     #(current['step'] < (current['num_steps'] - cache_dic['first_enhance'] + 1)):
#     #c_s < c_num_s - ca_f +1 
#     for i in range(cache_dic['max_order']):
#         if (cache_dic['cache'][-1][current['layer']][current['module']].get(i, None) is not None) and (current['step'] < (current['num_steps'] - cache_dic['first_enhance'] + 1)):
#             updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_dic['cache'][-1][current['layer']][current['module']][i]) / difference_distance
#         else:
#             break
    
#     cache_dic['cache'][-1][current['layer']][current['module']] = updated_taylor_factors

# def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor: 
#     """
#     Compute Taylor expansion error.
#     :param cache_dic: Cache dictionary.
#     :param current: Current step information.
#     """
#     x = current['step'] - current['activated_steps'][-1]
#     # x = current['t'] - current['activated_times'][-1]
#     output = 0

#     for i in range(len(cache_dic['cache'][-1][current['layer']][current['module']])):
#         output += (1 / math.factorial(i)) * cache_dic['cache'][-1][current['layer']][current['module']][i] * (x ** i)
    
#     return output

# def taylor_cache_init(cache_dic: Dict, current: Dict):
#     """
#     Initialize Taylor cache and expand storage for different-order derivatives.
#     :param cache_dic: Cache dictionary.
#     :param current: Current step information.
#     """
#     # if current['step'] == (current['num_steps'] - 1):
#     #     cache_dic['cache'][-1][current['layer']][current['module']] = {}
#     # else:
#     #     print("no taylor cache init")
        
        
        
    
#     # 只在第一步初始化
#     if current.get('step_idx', 0) == 0:
#         cache_dic['cache'][-1][current['layer']][current['module']] = {}
#     else:
#         A=1
#         #print("no taylor cache init")
        
def taylor_formula(cache_dic: Dict, current: Dict, order_limit: int | None = None) -> torch.Tensor:
    x = current['step'] - current['activated_steps'][-1]
    coeffs = cache_dic['cache'][-1][current['layer']][current['module']]  # {0: f, 1: f', 2: f'' ...}
    max_avail = len(coeffs) - 1  # 实际可用的最高阶（受是否已有足够 Full 的限制）
    use_up_to = max_avail if order_limit is None else min(max_avail, order_limit)

    out = 0
    for i in range(use_up_to + 1):
        out = out + (coeffs[i] * (x ** i) / math.factorial(i))
    return out

# def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
#     """
#     在 Full 步更新泰勒系数：0阶必存；若已有两个 Full 锚点，则给出 1阶（及以上，若 max_order>1）。
#     仅在 Full 步调用本函数；Taylor 步不要调用（只读缓存）。
#     """
#     # ---- 1) 健壮的嵌套字典保证 ----
#     step_cache = cache_dic['cache'][-1]                      # 当前轮缓存“页”
#     layer_cache = step_cache.setdefault(current['layer'], {}) 
#     mod_cache   = layer_cache.setdefault(current['module'], {})  # 形如 {0: f, 1: f', ...}

#     # ---- 2) 0阶系数：直接写入（建议在 Full 步写）----
#     # 注：feature 大多在推理时不需要 autograd，若担心图增长可 .detach()
#     mod_cache_0_prev = mod_cache.get(0, None)  # 先留一份旧的 0 阶，用于差分
#     mod_cache[0] = feature  # 也可 feature.detach() 视需要

#     # ---- 3) 一阶（及以上）差分：需要至少两个 Full 锚点 ----
#     hist = current.get('activated_steps', [])
#     if len(hist) < 2:
#         # 还不够两个 Full，直接返回（只有 0 阶）
#         return

#     # 关键：用“上一次 Full 的 t 减去这一次 Full 的 t”作为正的步长 h
#     # Diffusion 的 t 通常递减，所以 h = t_{prev} - t_{curr} > 0
#     h = float(hist[-2]) - float(hist[-1])
#     eps = torch.finfo(feature.dtype).eps if feature.is_floating_point() else 1e-12
#     if abs(h) < eps:
#         return  # 极端情况下步长为0，跳过高阶

#     # 仅当之前已有同阶系数时，才能递推更高阶
#     # 例如 i=0：需要 old f(t_prev)；i=1：需要 old f'(t_prev) ...
#     max_order = int(cache_dic.get('max_order', 1))
#     # 可选尾段门控（与原逻辑一致）
#     tail_guard = current['step'] < (current['num_steps'] - cache_dic.get('first_enhance', 0) + 1)
######BUG : 这个地方，mod_cache_0_prev没有用上  tail_guard似乎也得删除
#     updated = {0: mod_cache[0]}
#     for i in range(min(max_order, 8)):  # 上限给个小数，防御性
#         prev_i = mod_cache.get(i, None)   # 注意：此时 mod_cache 已经更新了 0 阶，但更高阶仍是“旧值”
#         if prev_i is not None and tail_guard:
#             # 后向差分形式：f^{(i+1)}(t_curr) ≈ (f^{(i)}(t_curr) - f^{(i)}(t_prev)) / h
#             updated[i+1] = (updated[i] - prev_i) / h
#         else:
#             break

#     # 覆盖写入（包含 0 阶与已算出的高阶）
#     layer_cache[current['module']] = updated
    


def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    修正后的版本：在 Full 步更新泰勒系数。
    """
    # ---- 1) 健壮的嵌套字典保证 ----
    step_cache = cache_dic['cache'][-1]
    layer_cache = step_cache.setdefault(current['layer'], {})
    mod_cache = layer_cache.setdefault(current['module'], {})

    # ---- 2) 先保存旧的系数，再写入新的0阶系数 ----
    old_coeffs = mod_cache.copy() # ★ 关键修正：复制一份旧的系数
    updated_coeffs = {0: feature} # 初始化新的系数词典，写入当前特征作为0阶

    # ---- 3) 一阶（及以上）差分：需要至少两个 Full 锚点 ----
    hist = current.get('activated_steps', [])
    if len(hist) < 2:
        # 还不够两个 Full，直接用新的0阶系数覆盖缓存
        layer_cache[current['module']] = updated_coeffs
        return

    h = float(hist[-2]) - float(hist[-1])
    eps = torch.finfo(feature.dtype).eps if feature.is_floating_point() else 1e-12
    if abs(h) < eps:
        layer_cache[current['module']] = updated_coeffs
        return

    max_order = int(cache_dic.get('max_order', 1))
    tail_guard = current['step'] < (current['num_steps'] - cache_dic.get('first_enhance', 0) + 1)

    # ---- 4) 修正后的循环，使用 old_coeffs 进行差分 ----
    # 计算一阶导数
    old_f0 = old_coeffs.get(0)
    if old_f0 is not None: #and tail_guard:
        # f'(t_curr) ≈ (f(t_curr) - f(t_prev)) / h
        updated_coeffs[1] = (updated_coeffs[0] - old_f0) / h

        # 递归计算更高阶导数
        for i in range(1, max_order):
            old_fi = old_coeffs.get(i)
            if old_fi is not None:
                 # f^(i+1)(t_curr) ≈ (f^i(t_curr) - f^i(t_prev)) / h
                updated_coeffs[i + 1] = (updated_coeffs[i] - old_fi) / h
            else:
                break # 如果旧的系数中没有更高阶，则停止

    # ---- 5) 用计算好的新系数覆盖缓存 ----
    layer_cache[current['module']] = updated_coeffs
 
def taylor_cache_init(cache_dic: Dict, current: Dict):
    step_cache  = cache_dic['cache'][-1]
    layer_cache = step_cache.setdefault(current['layer'], {})
    layer_cache.setdefault(current['module'], {})  # 不清空，不覆盖