from rich.console import Console

# 创建一个 Console 实例，你可以把它放在文件的顶部或者类的 __init__ 方法中
console = Console()
import time

# 1. 记录开始时间
start_total_program_time = time.perf_counter()

from typing import Callable, List, Type
import sys
sys.path.append('/')
import gymnasium as gym
import numpy as np
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
import argparse
import yaml
from scripts.maniskill_model import create_model, RoboticDiffusionTransformerModel
import torch
from collections import deque
from PIL import Image
import cv2
from pathlib import Path


from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.contrib import itertools as tqdm_it
from termcolor import colored, cprint
import random
from datetime import datetime
import inspect, sys, importlib, pathlib


from models.project_enums import PruningMode

from utils.profiler import global_profiler 



# =================================================================
# ========== 新增：为 Alpha vs Temp 数据生成 Facet Grid 的代码 ==========
# =================================================================

def plot_facet_grid_for_alpha_temp(df: pd.DataFrame, env_id: str, keep_tokens: int):
    """
    为包含 alpha 和 temp 参数的实验数据生成一个 Facet Grid 图表。

    Args:
        df (pd.DataFrame): 必须包含 'alpha', 'temp', 'mean_success_rate', 'std_dev' 列。
        env_id (str): 环境ID，用于图表标题。
        keep_tokens (int): 保留的token数量，用于图表标题。
    """
    
    # 1. 获取所有唯一的 alpha 值，以确定需要多少个子图
    unique_alphas = sorted(df['alpha'].unique())
    num_alphas = len(unique_alphas)
    
    if num_alphas == 0:
        print("DataFrame 中未找到 'alpha' 列，无法生成 Facet Grid。")
        return

    # 2. 动态计算子图的网格布局（例如，每行最多显示3个图）
    ncols = min(3, num_alphas)
    nrows = (num_alphas + ncols - 1) // ncols  # 向上取整

    # 3. 创建子图网格
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    
    # 将二维的 axes 数组展平为一维，方便迭代
    axes_flat = axes.flatten()

    # 4. 遍历每个 alpha 值，在对应的子图上绘图
    for i, alpha_val in enumerate(unique_alphas):
        ax = axes_flat[i]
        
        # 筛选出当前 alpha 值对应的数据
        subset_df = df[df['alpha'] == alpha_val]
        
        # 在当前子图(ax)上绘制误差棒图
        ax.errorbar(
            subset_df['temp'],
            subset_df['mean_success_rate'],
            yerr=subset_df['std_dev'],
            marker='o',
            linestyle='-',
            capsize=4
        )
        
        ax.set_title(f'Alpha = {alpha_val:.2f}', fontsize=12)
        ax.set_xlabel('Temperature (temp)', fontsize=10)
        ax.set_ylabel('Mean Success Rate (%)', fontsize=10)
        ax.grid(True, linestyle='--', linewidth=0.5)

    # 5. 如果子图数量少于网格大小，隐藏多余的空子图
    for j in range(num_alphas, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # 6. 添加一个统一的大标题
    fig.suptitle(
        f'Success Rate vs. Temperature (Faceted by Alpha)\n'
        f'Env: {env_id}, Tokens: {keep_tokens}/729',
        fontsize=16,
        weight='bold'
    )

    # 7. 自动调整布局，防止标题和标签重叠
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 8. 保存图像
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f'{env_id}_{keep_tokens}tokens_facet_grid_{now_str}.png'
    plt.savefig(image_filename, dpi=300)
    print(f"Facet Grid 图表已保存为: {image_filename}")
    plt.close(fig)




def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1", help=f"Environment to run motion planning solver on. ")
    parser.add_argument("-o", "--obs-mode", type=str, default="rgb", help="Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script.")
    parser.add_argument("-n", "--num-traj", type=int, default=25, help="Number of trajectories to test.")
    parser.add_argument("--only-count-success", action="store_true", help="If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Path to the pretrained model")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for the environment.")
    return parser.parse_args()

import random
import os

# set cuda 
args = parse_args()
# set random seeds
seed = args.random_seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

task2lang = {
    "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
    "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
    "StackCube-v1":  "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "PlugCharger-v1": "Pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.",
    "PushCube-v1": "Push and move a cube to a goal region in front of it."
}

env_id = args.env_id
env = gym.make(
    env_id,
    obs_mode=args.obs_mode,
    control_mode="pd_joint_pos",
    render_mode=args.render_mode,
    reward_mode="dense" if args.reward_mode is None else args.reward_mode,
    sensor_configs=dict(shader_pack=args.shader),
    human_render_camera_configs=dict(shader_pack=args.shader),
    viewer_camera_configs=dict(shader_pack=args.shader),
    sim_backend=args.sim_backend
)

config_path = 'configs/base.yaml'
with open(config_path, "r") as fp:
    config = yaml.safe_load(fp)
pretrained_text_encoder_name_or_path = "/root/autodl-tmp/ckpt/google/t5-v1_1-xxl"
pretrained_vision_encoder_name_or_path = "/root/autodl-tmp/ckpt/google/siglip-so400m-patch14-384"
pretrained_path = args.pretrained_path
policy = create_model(
    args=config, 
    dtype=torch.bfloat16,
    pretrained=pretrained_path,
    pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
    pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path
)

if os.path.exists(f'text_embed_{env_id}.pt'):
    text_embed = torch.load(f'text_embed_{env_id}.pt')
else:
    
    torch.cuda.synchronize()  # 等待当前设备上的所有流完成:contentReference[oaicite:0]{index=0}
    start = torch.cuda.Event(enable_timing=True)  # 允许计时:contentReference[oaicite:1]{index=1}
    end   = torch.cuda.Event(enable_timing=True)
    start.record()

    # ===== 这里放置你的 GPU 代码片段 =====
    text_embed = policy.encode_instruction(task2lang[env_id])
    # =====================================
    torch.cuda.synchronize()  # 等待流中所有内核和事件完成:contentReference[oaicite:2]{index=2}
    end.record()
    # 再次同步，确保结束事件已被记录
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)  # 返回单位为 ms:contentReference[oaicite:3]{index=3}
    from colorama import init
    from termcolor import colored
    init()  # 在 Windows 上初始化支持
    print(
        f"policy.encode_instruction GPU time: "
        + colored(f"{elapsed_ms:.3f}", "red")
        + " ms"
    )
    
    torch.save(text_embed, f'text_embed_{env_id}.pt')


print("test inter text encode time")

language_time=0
#calculat text embed time
for i in range(0,30):
    
    language_time=0
    torch.cuda.synchronize()  # 等待当前设备上的所有流完成:contentReference[oaicite:0]{index=0}
    start = torch.cuda.Event(enable_timing=True)  # 允许计时:contentReference[oaicite:1]{index=1}
    end   = torch.cuda.Event(enable_timing=True)
    start.record()

    # ===== 这里放置你的 GPU 代码片段 =====
    text_embed = policy.encode_instruction(task2lang[env_id])
    # =====================================
    torch.cuda.synchronize()  # 等待流中所有内核和事件完成:contentReference[oaicite:2]{index=2}
    end.record()
    # 再次同步，确保结束事件已被记录
    torch.cuda.synchronize()
    language_time = start.elapsed_time(end)  # 返回单位为 ms:contentReference[oaicite:3]{index=3}
    
    print(i)
    print(
        f"policy.encode_instruction GPU time: "
        + colored(f"{language_time:.3f}", "red")
        + " ms"
    )
    
    torch.save(text_embed, f'text_embed_{env_id}.pt')


MAX_EPISODE_STEPS = 400 
total_episodes = args.num_traj  
success_count = 0  

base_seed = 20241201
import tqdm
avg_avg_time_list=[]


lzw_log_path="/root/autodl-tmp/RoboticsDiffusionTransformer/lzw_logs/1.txt"

lzw_log_PATH = Path(lzw_log_path)  

#alpha=0.7, temp=0.31622776601683794
alpha= np.linspace(1, 0, num=3)
#alpha = [0.7]
temp = np.logspace(np.log10(0.01), np.log10(10.0), num=3)
#[0.01, 0.05623413, 0.31622777, 1.77827941, 10.]
#temp=[np.logspace(np.log10(0.01), np.log10(10.0), num=5)[3]]
print(temp)
cdp_para=np.linspace(1, 0, num=4)
# 2. 执行模型并收集结果
success_data = []
#729  182*1
keep_img_token_nums = 182
#pruning_mode = "no_img_token_pruning" #score
#pruning_mode ="score"

#pruning_mode = PruningMode.CDPRUNER    
pruning_mode = PruningMode.SCORE  # 使用枚举成员
#pruning_mode=PruningMode.RANDOM
#pruning_mode=PruningMode.NO_PRUNING
repeat_exp_num=10
for a, b in tqdm_it.product(alpha, temp,
                             desc='grid search'):
#for cdp_it in tqdm.trange(len(cdp_para)):
    rep_suc=[]
    for re_it in tqdm.trange(repeat_exp_num):
        # --- 标记一个新的 Experiment Run 开始 ---
        global_profiler.start_experiment_run(run_idx=re_it) # <--- 在这里调用新方法
        #success_rate=0
        success_count = 0  
        # --- 标记一个新的 Trial 开始 ---
        # --- 在这里重置种子，在每一次试验开始时 ---
        print(f"Running trial with alpha={a}, temp={b}. Re-seeding with {base_seed}.")
        #print(f"Running trial with cdp_para={cdp_para[cdp_it]},  Re-seeding with {base_seed}.")
        random.seed(base_seed)
        os.environ['PYTHONHASHSEED'] = str(base_seed)
        np.random.seed(base_seed)
        torch.manual_seed(base_seed)
        torch.cuda.manual_seed(base_seed)
    # 如果您使用多GPU，可能还需要 torch.cuda.manual_seed_all(base_script_seed)
        for episode in tqdm.trange(total_episodes):
            global_profiler.start_trial(trial_idx=episode) # <--- 在这里调用
            obs_window = deque(maxlen=2)
            obs, _ = env.reset(seed = episode + base_seed)
            policy.reset()
            policy.alpha = a
            policy.temp = b
            policy.keep_img_token_nums=keep_img_token_nums
            policy.vis_count=0
            policy.pruning_mode=pruning_mode
            policy.save_vis_doc_name=str(args.env_id)+str(policy.pruning_mode)+str(keep_img_token_nums)+'_729 token'
            #policy.cdp_para=cdp_para[cdp_it]
            policy.cdp_para=None
            
            img = env.render().squeeze(0).detach().cpu().numpy()
            obs_window.append(None)
            obs_window.append(np.array(img))
            proprio = obs['agent']['qpos'][:, :-1]

            global_steps = 0
            video_frames = []

            success_time = 0
            done = False
            
            while global_steps < MAX_EPISODE_STEPS and not done:
                image_arrs = []
                for window_img in obs_window:
                    image_arrs.append(window_img)
                    image_arrs.append(None)
                    image_arrs.append(None)
                images = [Image.fromarray(arr) if arr is not None else None
                        for arr in image_arrs]
                #action [64,8] 
                actions = policy.step(proprio, images, text_embed).squeeze(0).cpu().numpy()
                
                # Take 8 steps since RDT is trained to predict interpolated 64 steps(actual 14 steps)
                #[16,8]
                actions = actions[::4, :]
                
                for idx in range(actions.shape[0]):
                    action = actions[idx]
                    obs, reward, terminated, truncated, info = env.step(action)
                    img = env.render().squeeze(0).detach().cpu().numpy()
                    obs_window.append(img)
                    proprio = obs['agent']['qpos'][:, :-1]
                    video_frames.append(img)
                    global_steps += 1
                    if terminated or truncated:
                        assert "success" in info, sorted(info.keys())
                        if info['success']:
                            success_count += 1
                            done = True
                            break 
            print(f"Trial {episode+1} finished, success: {info['success']}, steps: {global_steps}")
            print(f"policy.alpha = {policy.alpha}"+f"policy.temp = {policy.temp}")
            
            
            
                
            
            #avg_diffuion_time = sum(policy.runing_time) / len(policy.runing_time)
            #print("avg_diffuion_time: ",avg_diffuion_time)  
            #content  = f"{str(policy.runing_time)}\n"    
            # precision = 3                                 # 想保留的小数位
            # formatted = [f"{t:.{precision}f}" for t in (policy.runing_time)]
            # content   = f"[{', '.join(formatted)}]\n"     
            # content += f"avg_diffuion_time={avg_diffuion_time:.3f}\n"           # => score=0.93   + 换行
            # with lzw_log_PATH.open('a', encoding='utf-8') as f:
            #     f.write(content)
            
            
            #avg_avg_time_list.append(avg_diffuion_time)
        #success_rate=8 +random.uniform(1.5, 2) 
        success_rate = success_count / total_episodes * 100
    #print(f"Success rate: {success_rate}%")
        print(f"policy.alpha = {policy.alpha}"+f"policy.temp = {policy.temp}")
        s = colored(f"Success rate: {success_rate}%", "yellow", attrs=["bold"])
        print(s)
        rep_suc.append(success_rate)
    print(f"rep Success rate: {rep_suc}%")
    # --- 当一组重复实验 (repeat_exp_num) 全部结束后 ---
    # 1. 将列表转换为 NumPy 数组以便高效计算
    rep_suc_array = np.array(rep_suc)
    #    注意: 当重复次数 > 1 时，这些计算才有意义
    if len(rep_suc_array) > 1:
        mean_rate = np.mean(rep_suc_array)
        # 使用样本标准差(ddof=1)更符合统计惯例，因为它能更好地估计总体标准差
        std_dev = np.std(rep_suc_array, ddof=1) 
        variance = np.var(rep_suc_array, ddof=1)
    else: # 如果只重复一次，标准差和方差为0
        mean_rate = rep_suc_array[0] if len(rep_suc_array) > 0 else 0
        std_dev = 0
        variance = 0
    
    print(f"参数组 (alpha={a}, temp={b}) 的结果:")
    #print(f"参数组 (cdp_para[cdp_it]={cdp_para[cdp_it]}) 的结果:")
    print(f"  - 原始成功率列表: {rep_suc}")
    cprint(f"  - 平均成功率 (Mean): {mean_rate:.2f}%", "cyan", attrs=["bold"])
    print(f"  - 标准差 (Std Dev): {std_dev:.2f}")
    print(f"  - 方差 (Variance): {variance:.2f}")
    # 4. 创建一个包含所有信息的字典
    # result_entry = {
        
    #     'cdp_para[cdp_it]': cdp_para[cdp_it],
    #     'mean_success_rate': mean_rate,
    #     'std_dev': std_dev,
    #     'variance': variance,
    #     'raw_success_rates': rep_suc  # 保存原始数据列表
    # }
    result_entry = {
            'alpha': a,
            'temp': b,
            'mean_success_rate': mean_rate,
            'std_dev': std_dev,
            'variance': variance,
            'raw_success_rates': rep_suc
        }
    success_data.append(result_entry)
    

    # avg_avg_diffuion_time = sum(avg_avg_time_list) / len(avg_avg_time_list)
    # precision = 3                                 # 想保留的小数位
    # formatted = [f"{t:.{precision}f}" for t in (avg_avg_time_list)]
    # content   = f"[{', '.join(formatted)}]\n" 
    # content += f"avg_avg_diffuion_time={avg_avg_diffuion_time}\n"           # => score=0.93   + 换行
    # with lzw_log_PATH.open('a', encoding='utf-8') as f:
    #     f.write(content)

    
    #print(f"avg_avg_diffuion_time: {avg_avg_diffuion_time}ms")
    
    
    
#old df for a b and score pruning  
df = pd.DataFrame(success_data)
# # 将原始数据列表也保存到 Excel，注意它会以字符串形式保存
df.to_excel('results_with_stats.xlsx', index=False)

# # ====================== 优化后的绘图代码 ======================
# # 将 DataFrame 保存到 Excel/CSV，建议文件名与图片文件名保持一致，方便追溯
# now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
# excel_filename = f'results_{args.env_id}_{keep_img_token_nums}tokens_{now_str}.xlsx'
# df.to_excel(excel_filename, index=False)
# print(f"实验结果已保存至: {excel_filename}")

# # --- 开始绘图 ---
# plt.figure(figsize=(12, 8)) # 创建一个更大、比例更好的画布

# # 设置一个更美观的绘图风格
# plt.style.use('seaborn-v0_8-whitegrid')

# # 遍历每个 alpha 值来绘制曲线
# for a_val in sorted(df['alpha'].unique()):
#     sub_df = df[df['alpha'] == a_val]
#     # 使用 errorbar 同时绘制均值线和标准差范围，这能更好地展示数据的稳定性
#     plt.errorbar(
#         sub_df['temp'], 
#         sub_df['mean_success_rate'], 
#         yerr=sub_df['std_dev'],  # yerr 参数用来显示标准差
#         marker='o',             # 数据点样式
#         linestyle='-',          # 线条样式
#         capsize=4,              # 误差棒顶部的横线宽度
#         label=f'alpha={a_val}'
#     )

# # --- 设置图表的各种标签和标题 ---
# plt.xlabel('Temperature (temp)', fontsize=12)
# plt.ylabel('Mean Success Rate (%)', fontsize=12)
# # 使用 f-string 动态生成标题，更清晰
# title = (
#     f"Success Rate vs. Temperature for different Alpha values\n"
#     f"Env: {args.env_id}, Pruning: {policy.pruning_mode.name}, Tokens: {keep_img_token_nums}/729"
# )
# plt.title(title, fontsize=14, weight='bold')
# plt.legend(title='Alpha Values', fontsize=10) # 给图例也加上标题
# plt.grid(True, which='both', linestyle='--', linewidth=0.5) # 添加网格线

# plt.tight_layout() # 自动调整布局，防止标签重叠

# # --- 【关键步骤】保存图像文件 ---
# # 使用动态文件名，包含环境ID和时间戳，避免覆盖
# image_filename = f'{args.env_id}_{keep_img_token_nums}tokens_{now_str}_{pruning_mode.name}.png'
# plt.savefig(image_filename, dpi=300) # dpi=300 可以获得高分辨率图像
# print(f"绘图已保存为: {image_filename}")

# # ... 保存图片的代码不变 ...

# s = colored(str(success_data), "yellow", attrs=["bold"])
# print(s)

# # 按平均成功率找到最佳结果
# best_rate = max(item['mean_success_rate'] for item in success_data)

# # 打印所有等于最大值的项，并显示更丰富的信息
# for it in success_data:
#     if it['mean_success_rate'] == best_rate:
#         print(colored("="*20 + " BEST RESULT " + "="*20, "green"))
#         print(f"  - Parameters: alpha={it['alpha']}, temp={it['temp']}")
#         print(f"  - Mean Success Rate: {it['mean_success_rate']:.2f}%")
#         print(f"  - Standard Deviation: {it['std_dev']:.2f}")
#         print(f"  - Raw Data: {it['raw_success_rates']}")
#         print(colored("="*53, "green"))

# print("policy.keep_img_token_nums:",policy.keep_img_token_nums)
# print("policy.save_vis_doc_name:",policy.save_vis_doc_name)
# print("policy.pruning_mode:",policy.pruning_mode)
# print("repeat_exp_num: ",repeat_exp_num)
# now = datetime.now()

# # 格式化为 年日月小时分
# print(now.strftime("%Y-%d-%m %H:%M"))
# # 2. 记录结束时间
# end_time = time.perf_counter()

# # 3. 计算并打印流逝时间
# elapsed_time = end_time - start_total_program_time
# print(f"程序总共耗时: {elapsed_time:.4f} 秒")

##########################


# ====================== 优化后的绘图代码 (针对CDP参数) ======================
# 将 DataFrame 保存到 Excel/CSV
now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
excel_filename = f'results_{args.env_id}_{pruning_mode.name}_{keep_img_token_nums}tokens_{now_str}.xlsx'
df.to_excel(excel_filename, index=False)
print(f"实验结果已保存至: {excel_filename}")


# ====================== 修正后的“最佳结果”打印部分 ======================
s = colored(str(success_data), "yellow", attrs=["bold"])
print(s)

# 按平均成功率找到最佳结果
# 注意：如果成功率有多个并列最高，这里会打印所有并列最高的项
best_rate = df['mean_success_rate'].max()
best_results_df = df[df['mean_success_rate'] == best_rate]
plot_facet_grid_for_alpha_temp(df, args.env_id, keep_img_token_nums)
print(colored("\n" + "="*20 + " BEST RESULT(S) " + "="*20, "green"))
for index, row in best_results_df.iterrows():
    # MODIFIED: 打印正确的参数
    print(f"  - Parameters: alpha={row['alpha']:.3f}, temp={row['temp']:.4f}")
    print(f"  - Mean Success Rate: {row['mean_success_rate']:.2f}%")
    print(f"  - Standard Deviation: {row['std_dev']:.2f}")
    print(f"  - Raw Data: {row['raw_success_rates']}")
    print("-" * 54)
print(colored("="*54, "green"))


print("policy.keep_img_token_nums:",policy.keep_img_token_nums)
print("policy.save_vis_doc_name:",policy.save_vis_doc_name)
print("policy.pruning_mode:",policy.pruning_mode)
print("repeat_exp_num: ",repeat_exp_num)
now = datetime.now()

# 格式化为 年日月小时分
print(now.strftime("%Y-%d-%m %H:%M"))
# 2. 记录结束时间
end_time = time.perf_counter()

# 3. 计算并打印流逝时间
elapsed_time = end_time - start_total_program_time
print(f"程序总共耗时: {elapsed_time:.4f} 秒")
# =================================================================
# ========== 在所有大循环结束后，在这里进行最终性能总结 ==========
# =================================================================

print("\n" + "="*80)
print("🏁 所有实验已完成，正在生成最终性能总结报告...")
print("="*80)

# 1. 生成报告
# 这个函数会处理所有收集到的数据，并计算出总体平均值和标准差
final_report = global_profiler.generate_report()

# 2. 以易于阅读的方式显示报告
# 您需要的所有信息（总体均值、标准差）都会在这里清晰地打印出来
global_profiler.display_report(final_report)

# 3. (可选) 如果您想在代码中直接使用这些值
if "overall_summary" in final_report and final_report["overall_summary"]:
    overall_stats = final_report["overall_summary"]
    print("\n--- 提取关键性能指标 ---")
    
    if "Vision" in overall_stats:
        vision_stats = overall_stats["Vision"]
        print(f"👁️ Vision 时间:")
        print(f"  - 总体平均值 (Mean): {vision_stats['mean_of_means']:.2f} ms")
        print(f"  - 总体标准差 (Std Dev): {vision_stats['std_of_means']:.2f} ms")
    
    if "Action" in overall_stats:
        action_stats = overall_stats["Action"]
        print(f"🏃 Action 时间:")
        print(f"  - 总体平均值 (Mean): {action_stats['mean_of_means']:.2f} ms")
        print(f"  - 总体标准差 (Std Dev): {action_stats['std_of_means']:.2f} ms")
    # 这里的 count 是您运行的总 run 次数 (4 * 20 = 80)
    print(f"\n* 以上统计数据基于全部 {overall_stats['Vision']['count']} 次独立实验运行 (experiment run) 的结果。")