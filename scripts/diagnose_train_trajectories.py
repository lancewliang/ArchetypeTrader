"""分析 k=8 在 train DP 轨迹上的分布

看 k=8 对应的轨迹：
1. 动作分布（flat 比例）
2. 轨迹总 reward
3. 对应的市场特征（价格方向、波动率）
"""
import sys, numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, ".")

from src.config import parse_args
from src.evaluation.model_loader import load_phase1_model
from src.data.dataset import TrajectoryDataset
from src.phase1.vq_encoder import VQEncoder

config = parse_args(["--train-batch-id", "batch_001"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

codebook, decoder, normalizer = load_phase1_model(config, "AL", device)

# 加载 encoder
import os, torch
ckpt = torch.load(
    "result/AL/batch_001/phase1_archetype_discovery/AL_vq_model.pt",
    map_location=device, weights_only=False,
)
encoder = VQEncoder(
    state_dim=config.state_dim, action_dim=config.action_dim,
    latent_dim=config.latent_dim, hidden_dim=config.lstm_hidden_dim,
).to(device)
encoder.load_state_dict(ckpt["encoder"])
encoder.eval()

# 加载轨迹（归一化版本，与训练一致）
ds = TrajectoryDataset.from_npz(
    "result/AL/batch_001/dp_trajectories/trajectories.npz", normalize=True,
)

# 同时加载原始轨迹（未归一化）用于 reward/action 分析
traj_raw = np.load(
    "result/AL/batch_001/dp_trajectories/trajectories.npz", allow_pickle=True,
)
raw_actions = traj_raw["actions"]   # (N, 72) int32
raw_rewards = traj_raw["rewards"]   # (N, 72) float64
print("批量编码训练轨迹...")
all_codes = []
N = len(ds)
batch_size = 1024
with torch.no_grad():
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        # 直接切片 tensor（Dataset 内部已是 Tensor）
        s_t = ds.states[i:end].to(device)
        a_t = ds.actions[i:end].to(device)
        r_t = ds.rewards[i:end].to(device)
        z_e = encoder(s_t, a_t, r_t)
        _, code_idx, _ = codebook.quantize(z_e)
        all_codes.append(code_idx.cpu().numpy())

all_codes = np.concatenate(all_codes)   # (N,)

N = len(all_codes)
print(f"编码完成: {N} 条轨迹\n")

K = config.num_archetypes

# ---- 1. 各 archetype 在 train 上的分布 ----
print("【1】各 archetype 在 train DP 轨迹上的分布")
print(f"{'k':>4}  {'count':>7}  {'freq%':>7}  {'avg_reward':>12}  {'flat%':>8}  {'long%':>8}  {'short%':>8}")
print("-" * 65)
for k in range(K):
    mask = all_codes == k
    cnt = mask.sum()
    if cnt == 0:
        print(f"{k:>4}  {0:>7}  {0:>7.1f}%  {'N/A':>12}")
        continue
    acts = raw_actions[mask]          # (cnt, 72)
    rews = raw_rewards[mask].sum(-1)  # (cnt,) 每条轨迹总 reward
    flat_pct  = (acts == 1).mean() * 100
    long_pct  = (acts == 2).mean() * 100
    short_pct = (acts == 0).mean() * 100
    avg_rew = rews.mean()
    print(f"{k:>4}  {cnt:>7}  {cnt/N*100:>7.1f}%  {avg_rew:>12.2f}  {flat_pct:>7.1f}%  {long_pct:>7.1f}%  {short_pct:>7.1f}%")

# ---- 2. k=8 详细分析 ----
k8_mask = all_codes == 8
k8_acts = raw_actions[k8_mask]
k8_rews = raw_rewards[k8_mask].sum(-1)

print(f"\n【2】k=8 详细分析 (n={k8_mask.sum()})")
print(f"  总 reward 分布:")
print(f"    均值: {k8_rews.mean():.2f}")
print(f"    中位数: {np.median(k8_rews):.2f}")
print(f"    正收益比例: {(k8_rews > 0).mean()*100:.1f}%")
print(f"    零收益(|r|<1)比例: {(np.abs(k8_rews) < 1).mean()*100:.1f}%")

# 动作序列模式
all_flat = (k8_acts == 1).all(axis=1)
print(f"\n  动作序列模式:")
print(f"    全程 flat (72步全是1): {all_flat.sum()} ({all_flat.mean()*100:.1f}%)")
print(f"    flat 步数 >= 70: {((k8_acts == 1).sum(axis=1) >= 70).sum()}")
print(f"    flat 步数 >= 60: {((k8_acts == 1).sum(axis=1) >= 60).sum()}")
print(f"    flat 步数 < 50:  {((k8_acts == 1).sum(axis=1) < 50).sum()}")

# ---- 3. k=8 vs 其他 archetype 的 reward 对比 ----
print(f"\n【3】各 archetype 平均 reward 对比 (按 avg_reward 排序)")
k_stats = []
for k in range(K):
    mask = all_codes == k
    if mask.sum() == 0:
        continue
    rews = raw_rewards[mask].sum(-1)
    k_stats.append((k, rews.mean(), rews.std(), mask.sum()))
k_stats.sort(key=lambda x: x[1])
print(f"{'k':>4}  {'avg_reward':>12}  {'std':>10}  {'n':>7}")
print("-" * 40)
for k, avg, std, n in k_stats:
    marker = " ← k=8" if k == 8 else ""
    print(f"{k:>4}  {avg:>12.2f}  {std:>10.2f}  {n:>7}{marker}")

# ---- 4. 关键问题：k=8 在 train 上是否真的是"低 reward"轨迹 ----
print(f"\n【4】k=8 在 train 上的 reward 是否本来就低？")
other_rews = raw_rewards[~k8_mask].sum(-1)
print(f"  k=8  平均 reward: {k8_rews.mean():.2f}  (n={k8_mask.sum()})")
print(f"  其他 平均 reward: {other_rews.mean():.2f}  (n={(~k8_mask).sum()})")
ratio = k8_rews.mean() / (other_rews.mean() + 1e-8)
print(f"  比值: {ratio:.3f}")
if ratio < 0.5:
    print("  → k=8 在 train 上本来就是低收益轨迹，DP 在这些 horizon 上也赚不到钱")
    print("    说明 k=8 代表的是'市场本身无利可图'的行情，不是 selector 的问题")
elif ratio > 0.8:
    print("  → k=8 在 train 上 reward 和其他 archetype 差不多")
    print("    说明 k=8 在 train 上是有效的，问题是 val 上泛化失败")
else:
    print("  → k=8 在 train 上 reward 偏低，但不是最差")
