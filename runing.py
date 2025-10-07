import subprocess, re, statistics, sys
from tqdm import tqdm           # pip install tqdm

CMD = [
    "xvfb-run", "-s", "-screen 0 1920x1080x24",
    "python", "-m", "eval_sim.eval_rdt_maniskill",
    "--pretrained_path", "/root/autodl-tmp/ckpt/rdt/maniskill-model/rdt/mp_rank_00_model_states.pt",
    "--e", "PickCube-v1",
]

# 两条正则分别提取 success rate 和 diffusion time
PAT_SUCC = re.compile(r"Success rate:\s*([0-9.]+)")
PAT_TIME = re.compile(r"avg_avg_diffuion_time:\s*([0-9.]+)")

succ_rates, times = [], []

for i in tqdm(range(6), desc="100 runs"):
    res = subprocess.run(CMD, text=True, capture_output=True, check=True)

    # —— 成功率 —— #
    m1 = PAT_SUCC.search(res.stdout)
    if not m1:
        print(f"[Run {i}] 找不到 Success rate 行，输出如下：\n{res.stdout}", file=sys.stderr)
        sys.exit(1)
    succ_rates.append(float(m1.group(1)))

    # —— 扩散耗时 —— #
    m2 = PAT_TIME.search(res.stdout)
    if not m2:
        print(f"[Run {i}] 找不到 avg_avg_diffuion_time 行，输出如下：\n{res.stdout}", file=sys.stderr)
        sys.exit(1)
    times.append(float(m2.group(1)))

# 计算统计量
succ_mean = statistics.mean(succ_rates)
succ_std  = statistics.stdev(succ_rates)      # 样本标准差 (ddof=1)
time_mean = statistics.mean(times)
time_std  = statistics.stdev(times)

print("\n====== 统计结果 ======")
print(f"Success rate -> mean={succ_mean:.2f}%   std={succ_std:.2f}")
print(f"Diffusion time -> mean={time_mean:.2f} ms   std={time_std:.2f} ms")