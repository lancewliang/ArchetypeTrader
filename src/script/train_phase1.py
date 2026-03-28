#!/usr/bin/env python
"""Phase I 训练脚本 — 原型发现

# 需求: 7.1, 4.6, 4.7, 4.8, 7.5, 7.6, 7.7
#
# 流程:
# 1. 加载特征数据，初始化 TradingEnv
# 2. 调用 DPPlanner 生成 30k 示范轨迹并保存
# 3. 初始化 VQ Encoder、Codebook、Decoder
# 4. 训练 100 epochs
#    损失函数 L = L_rec + ||sg[z_e] - z_q||² + 0.25 × ||z_e - sg[z_q]||²
# 5. 保存模型到 result/phase1_archetype_discovery/
# 6. 输出训练日志（loss 曲线）
#
# 本版本说明:
# 1. 回归论文 Phase I 的核心结构：encoder -> codebook -> decoder。
# 2. 移除上一版额外加入的 usage balance / reward consistency / warmup 等工程项，
#    先验证严格论文配方本身是否能够学出可用 archetype。
# 3. Decoder 回归 p_theta_d(a_demo | s_demo, z_q) 的论文接口，
#    训练时不再依赖 previous action 的 teacher forcing shortcut。
# 4. 轨迹生成改为“按 horizon 采样 chunk 再跑 DP”，更贴近论文描述；
#    同时显式保存 horizon_indices，避免验证阶段把 sampled chunk 错配回错误 horizon。
# 5. 训练结束后自动执行 validation，直接落盘 Phase I 报告。
#
# 本轮新增（仍保持论文框架不变）:
# 1. 先按 horizon 不放回覆盖，再在需要时开启下一轮洗牌，避免一开始就有放回采样。
# 2. 对完全重复的 demo trajectory 做去重，降低 D 中重复 chunk 对单一 code 的挤压。
# 3. 在“论文的 sample n chunks”语义内，按 trade type / entry / exit / holding / return bucket
#    做均衡抽样，提升训练集 D 中 archetype 覆盖面，但不改 DP / Encoder / VQ / Decoder 结构。
# 4. 落盘 trajectory audit 报告，方便定位 duplicate_sample_ratio 和行为分布是否仍然异常。
#
# 论文关联:
# - Section 4.1: VQ encoder / codebook / decoder
# - Eq. (4): L = L_rec + ||sg[z_e] - z_q||² + beta_0 * ||z_e - sg[z_q]||²
# - Algorithm 1: 单次交易约束动态规划生成 demo trajectories
#
# 用法:
#   python scripts/train_phase1.py --pair BTC
#   python scripts/train_phase1.py --pair ETH --phase1-epochs 50 --batch-size 128
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import parse_args
from src.data.dataset import TrajectoryDataset
from src.data.feature_pipeline import FeaturePipeline
from src.env.trading_env import TradingEnv
from src.phase1.codebook import VQCodebook
from src.phase1.dp_planner import DPPlanner
from src.phase1.validation import ValidationConfig, run_phase1_validation, save_validation_report
from src.phase1.vq_decoder import VQDecoder
from src.phase1.vq_encoder import VQEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)

FLAT_ACTION = 1


def resolve_traj_path(result_dir: str, pair: str) -> str:
    """兼容两种轨迹命名方式，避免训练脚本与 DP 保存文件名不一致。"""
    candidates = [
        os.path.join(result_dir, pair, "dp_trajectories", "trajectories.npz"),
        os.path.join(result_dir, pair, "dp_trajectories", f"{pair}_trajectories.npz"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def load_trajectory_arrays(traj_path: str) -> Dict[str, np.ndarray]:
    data = np.load(traj_path)
    arrays: Dict[str, np.ndarray] = {
        "states": data["states"],
        "actions": data["actions"],
        "rewards": data["rewards"],
    }
    # 论文对齐版新增：保存每条 demo 实际来自哪个 horizon，
    # 让 validation 在 sampled chunk 场景下仍能正确回放 reward。
    if "horizon_indices" in data:
        arrays["horizon_indices"] = data["horizon_indices"]
    return arrays


def save_trajectory_arrays(traj_path: str, arrays: Dict[str, np.ndarray]) -> None:
    """保存 demo 轨迹到 npz。"""
    os.makedirs(os.path.dirname(traj_path), exist_ok=True)
    np.savez(
        traj_path,
        states=arrays["states"],
        actions=arrays["actions"],
        rewards=arrays["rewards"],
        horizon_indices=arrays["horizon_indices"],
    )


def save_trajectory_audit(audit_path: str, audit: Dict[str, Any]) -> None:
    """保存轨迹去重与抽样审计报告。"""
    os.makedirs(os.path.dirname(audit_path), exist_ok=True)
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)


def _sequence_hash(states: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> str:
    """对完整 demo trajectory 做稳定哈希，用于精确去重。"""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(states).tobytes())
    h.update(np.ascontiguousarray(actions).tobytes())
    h.update(np.ascontiguousarray(rewards).tobytes())
    return h.hexdigest()


def _build_visit_order(num_horizons: int, target_count: int, seed: int) -> np.ndarray:
    """构造 horizon 访问顺序：优先不放回覆盖，再按轮次重新洗牌。

    这样仍属于论文“sample n chunks”的实现范畴，但能明显降低早期重复。
    """
    if num_horizons <= 0:
        raise ValueError("num_horizons 必须 > 0")
    if target_count <= 0:
        raise ValueError("target_count 必须 > 0")

    rng = np.random.default_rng(seed)
    visit_order: List[int] = []
    while len(visit_order) < target_count:
        visit_order.extend(rng.permutation(num_horizons).tolist())
    return np.asarray(visit_order[:target_count], dtype=np.int32)


def _extract_trade_profile(actions: np.ndarray, rewards: np.ndarray, horizon: int) -> Dict[str, int | float | str]:
    """从 demo 序列中抽取“论文内可解释”的行为摘要。"""
    non_flat = np.flatnonzero(actions != FLAT_ACTION)
    total_return = float(np.sum(rewards))

    if non_flat.size == 0:
        return {
            "trade_type": "flat",
            "entry_idx": -1,
            "exit_idx": -1,
            "holding_length": 0,
            "total_return": total_return,
        }

    entry_idx = int(non_flat[0])
    exit_idx = int(non_flat[-1])
    holding_length = int(exit_idx - entry_idx + 1)

    first_trade_action = int(actions[entry_idx])
    if first_trade_action == 0:
        trade_type = "short"
    elif first_trade_action == 2:
        trade_type = "long"
    else:
        trade_type = "flat"

    return {
        "trade_type": trade_type,
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "holding_length": holding_length,
        "total_return": total_return,
    }


def _time_bucket(index: int, horizon: int, num_buckets: int = 4) -> str:
    """把开/平仓位置离散到少量时间桶，避免桶空间过碎。"""
    if index < 0:
        return "none"
    clipped = min(max(index, 0), horizon - 1)
    bucket = int((clipped * num_buckets) / max(horizon, 1))
    bucket = min(bucket, num_buckets - 1)
    return f"t{bucket}"


def _holding_bucket(length: int, horizon: int) -> str:
    """把持仓长度离散成少量宽桶。"""
    if length <= 0:
        return "h0"
    ratio = float(length) / float(max(horizon, 1))
    if ratio <= 0.10:
        return "h1"
    if ratio <= 0.25:
        return "h2"
    if ratio <= 0.50:
        return "h3"
    return "h4"


def _quantile_bucket_ids(values: np.ndarray, num_buckets: int = 5) -> np.ndarray:
    """按 return 分位数分桶；若返回分布退化，则退回单桶。"""
    if values.size == 0:
        return np.zeros((0,), dtype=np.int32)
    if np.allclose(values, values[0]):
        return np.zeros(values.shape[0], dtype=np.int32)

    quantiles = np.linspace(0.0, 1.0, num_buckets + 1)
    edges = np.quantile(values, quantiles)
    edges = np.asarray(edges, dtype=np.float64)

    # 去掉重复边界，避免 digitize 产生空洞。
    unique_edges = np.unique(edges)
    if unique_edges.size <= 2:
        return np.zeros(values.shape[0], dtype=np.int32)

    inner_edges = unique_edges[1:-1]
    bucket_ids = np.digitize(values, inner_edges, right=False)
    return bucket_ids.astype(np.int32)


def _counter_to_dict(counter: Counter[str]) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda item: item[0])}


def _bucket_name(profile: Dict[str, int | float | str], return_bucket: int, horizon: int) -> str:
    """构造均衡抽样桶名。

    这一步不改变论文中的训练目标，只是让 sample n chunks 得到的 D 更均衡，
    使离散 archetype 更有机会被多个 code 表达出来。
    """
    trade_type = str(profile["trade_type"])
    entry_bucket = _time_bucket(int(profile["entry_idx"]), horizon=horizon)
    exit_bucket = _time_bucket(int(profile["exit_idx"]), horizon=horizon)
    hold_bucket = _holding_bucket(int(profile["holding_length"]), horizon=horizon)
    return f"{trade_type}|{entry_bucket}|{exit_bucket}|{hold_bucket}|r{return_bucket}"


def _select_balanced_indices(records: Sequence[Dict[str, Any]], target_count: int, seed: int) -> List[int]:
    """对去重后的 demo 做桶级 round-robin 选样。"""
    if target_count <= 0:
        return []
    if not records:
        return []

    rng = np.random.default_rng(seed)
    bucket_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        bucket_to_indices[str(record["bucket"])].append(idx)

    for bucket_indices in bucket_to_indices.values():
        rng.shuffle(bucket_indices)

    bucket_names = list(bucket_to_indices.keys())
    rng.shuffle(bucket_names)

    selected: List[int] = []
    exhausted = False
    while len(selected) < min(target_count, len(records)) and (not exhausted):
        exhausted = True
        for bucket_name in bucket_names:
            bucket_indices = bucket_to_indices[bucket_name]
            if not bucket_indices:
                continue
            exhausted = False
            selected.append(bucket_indices.pop())
            if len(selected) >= min(target_count, len(records)):
                break

    # 若所有桶 round-robin 后仍未达到目标（理论上很少发生），
    # 用剩余未选中的样本随机补齐，但不引入重复。
    if len(selected) < min(target_count, len(records)):
        remaining = [idx for idx in range(len(records)) if idx not in set(selected)]
        rng.shuffle(remaining)
        need = min(target_count, len(records)) - len(selected)
        selected.extend(remaining[:need])

    return selected


def generate_paper_aligned_trajectories(
    planner: DPPlanner,
    env: TradingEnv,
    num_trajectories: int,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """按论文描述采样固定长度 chunk，并对每个 chunk 独立跑 DP。

    关键点：
    - 论文描述是“sample n data chunks of fixed length h”，而不是遍历一遍后循环复用。
    - 为了让后续 validation 能正确 reward replay，这里额外记录每条样本对应的 horizon_idx。
    - 本轮只在论文允许的“数据集 D 构造”层做改进：先不放回覆盖、精确去重、桶级均衡选样。
    """
    if env.num_horizons <= 0:
        raise ValueError("环境中没有可用 horizon，无法生成论文对齐版 demo trajectories。")
    if env.states_dataframe is None:
        raise ValueError("generate_paper_aligned_trajectories 需要 env.states_dataframe 不能为 None。")

    visit_order = _build_visit_order(num_horizons=env.num_horizons, target_count=num_trajectories, seed=seed)

    raw_records: List[Dict[str, Any]] = []
    logger.info(
        "开始按论文方式采样 DP 示范轨迹: num_horizons=%d, target_num_trajectories=%d, seed=%d",
        env.num_horizons,
        num_trajectories,
        seed,
    )

    for sample_idx, horizon_idx in enumerate(tqdm(visit_order.tolist(), desc="生成Paper-DP轨迹")):
        start = int(horizon_idx) * env.horizon
        end = start + env.horizon
        h_states = env.states_dataframe[start:end]
        price_end = min(end + 1, len(env.prices))
        h_prices = env.prices[start:price_end]

        s_demo, a_demo, r_demo = planner.plan(h_states, h_prices)
        record = {
            "horizon_idx": int(horizon_idx),
            "states": s_demo.astype(np.float32, copy=False),
            "actions": a_demo.astype(np.int32, copy=False),
            "rewards": r_demo.astype(np.float32, copy=False),
        }
        record.update(_extract_trade_profile(record["actions"], record["rewards"], horizon=env.horizon))
        record["hash"] = _sequence_hash(record["states"], record["actions"], record["rewards"])
        raw_records.append(record)

        if sample_idx == 0:
            logger.info(
                "Sample 0 shapes: horizon_idx=%d, s_demo=%s, a_demo=%s, r_demo=%s",
                horizon_idx,
                s_demo.shape,
                a_demo.shape,
                r_demo.shape,
            )

    # 先做精确去重，避免同一个 chunk 被多次采到时反复挤压 codebook。
    unique_records: List[Dict[str, Any]] = []
    first_seen_by_hash: Dict[str, int] = {}
    raw_trade_counter = Counter(str(record["trade_type"]) for record in raw_records)

    for raw_idx, record in enumerate(raw_records):
        record_hash = str(record["hash"])
        if record_hash in first_seen_by_hash:
            continue
        first_seen_by_hash[record_hash] = raw_idx
        unique_records.append(record)

    duplicate_count = len(raw_records) - len(unique_records)

    # 在去重后的集合上，按 return 分位数补充桶信息，再做均衡抽样。
    unique_returns = np.asarray([float(record["total_return"]) for record in unique_records], dtype=np.float64)
    return_bucket_ids = _quantile_bucket_ids(unique_returns, num_buckets=5)

    unique_trade_counter = Counter()
    bucket_counter = Counter()
    for idx, record in enumerate(unique_records):
        return_bucket = int(return_bucket_ids[idx]) if idx < return_bucket_ids.shape[0] else 0
        record["return_bucket"] = return_bucket
        record["bucket"] = _bucket_name(record, return_bucket=return_bucket, horizon=env.horizon)
        unique_trade_counter[str(record["trade_type"])] += 1
        bucket_counter[str(record["bucket"])] += 1

    selected_indices = _select_balanced_indices(records=unique_records, target_count=num_trajectories, seed=seed)
    selected_records = [unique_records[idx] for idx in selected_indices]
    selected_trade_counter = Counter(str(record["trade_type"]) for record in selected_records)
    selected_bucket_counter = Counter(str(record["bucket"]) for record in selected_records)

    arrays = {
        "states": np.asarray([record["states"] for record in selected_records], dtype=np.float32),
        "actions": np.asarray([record["actions"] for record in selected_records], dtype=np.int32),
        "rewards": np.asarray([record["rewards"] for record in selected_records], dtype=np.float32),
        "horizon_indices": np.asarray([record["horizon_idx"] for record in selected_records], dtype=np.int32),
    }

    audit: Dict[str, Any] = {
        "requested_num_trajectories": int(num_trajectories),
        "raw_num_trajectories": int(len(raw_records)),
        "num_unique_after_exact_dedup": int(len(unique_records)),
        "num_selected_for_training": int(len(selected_records)),
        "duplicate_count_removed": int(duplicate_count),
        "duplicate_ratio_before_dedup": float(duplicate_count / max(len(raw_records), 1)),
        "selection_was_capped_by_unique_count": bool(len(selected_records) < num_trajectories),
        "num_horizons": int(env.num_horizons),
        "unique_horizons_visited_before_dedup": int(len(set(int(v) for v in visit_order.tolist()))),
        "unique_horizons_selected_after_dedup": int(len(set(int(v) for v in arrays["horizon_indices"].tolist()))),
        "raw_trade_type_histogram": _counter_to_dict(raw_trade_counter),
        "unique_trade_type_histogram": _counter_to_dict(unique_trade_counter),
        "selected_trade_type_histogram": _counter_to_dict(selected_trade_counter),
        "unique_bucket_histogram_top20": dict(selected_bucket_counter.most_common(20)),
        "candidate_bucket_histogram_top20": dict(bucket_counter.most_common(20)),
        "selected_return_mean": float(np.mean([record["total_return"] for record in selected_records])) if selected_records else 0.0,
        "selected_return_std": float(np.std([record["total_return"] for record in selected_records])) if selected_records else 0.0,
    }

    logger.info(
        "轨迹审计: raw=%d, unique=%d, selected=%d, duplicate_ratio=%.4f, selected_trade_hist=%s",
        len(raw_records),
        len(unique_records),
        len(selected_records),
        audit["duplicate_ratio_before_dedup"],
        audit["selected_trade_type_histogram"],
    )

    return arrays, audit


def train_phase1_with_config(config: Any) -> None:
    """训练一个 Phase I 任务。

    单独拆出该函数，便于后续在不改论文结构的前提下做 beta_0 / K sweep。
    """
    # ----------------------------------------------------------------
    # Step 0: 解析配置
    # ----------------------------------------------------------------
    pair = config.pairs[0]  # 单交易对训练
    logger.info("Phase I 训练开始: pair=%s", pair)
    logger.info(
        "超参数: epochs=%d, batch_size=%d, lr=%.1e, latent_dim=%d, "
        "num_archetypes=%d, num_trajectories=%d, vq_beta0=%.2f",
        config.phase1_epochs,
        config.batch_size,
        config.learning_rate,
        config.latent_dim,
        config.num_archetypes,
        config.num_trajectories,
        config.vq_beta0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)

    # 默认强制重新生成 trajectories，
    # 避免继续沿用旧版高重复率轨迹文件，导致看起来“换了训练代码但数据没变”。
    force_regenerate_trajectories = getattr(config, "phase1_force_regenerate_trajectories", True)
    trajectory_seed = int(getattr(config, "seed", 42))
    validation_replay_abs_tol = getattr(config, "phase1_validation_replay_abs_tol", 1e-5)

    logger.info(
        "论文对齐配置: force_regenerate_trajectories=%s, trajectory_seed=%d",
        force_regenerate_trajectories,
        trajectory_seed,
    )

    # ----------------------------------------------------------------
    # Step 1: 加载特征数据，初始化 TradingEnv
    # ----------------------------------------------------------------
    logger.info("加载特征数据: data_dir=%s, pair=%s", config.data_dir, pair)
    pipeline = FeaturePipeline(config.data_dir, pair)
    train_df, _, _ = pipeline.get_state_vector()
    train_prices_df, _, _ = pipeline.get_prices()

    train_states = train_df.to_numpy()
    prices = train_prices_df["close"].to_numpy()

    logger.info(
        "训练集: states shape=%s, prices shape=%s",
        train_states.shape,
        prices.shape,
    )

    env = TradingEnv(
        states=train_states,
        prices=prices,
        pair=pair,
        horizon=config.horizon,
        states_dataframe=train_df,
    )

    logger.info(
        "TradingEnv 初始化完成 num_horizons = 总行数/切片内行数 = train_states.shape[0]/horizon: num_horizons=%d, horizon=%d",
        env.num_horizons,
        config.horizon,
    )

    # ----------------------------------------------------------------
    # Step 2: 生成 DP 示范轨迹
    # ----------------------------------------------------------------
    traj_path = resolve_traj_path(config.result_dir, pair)
    traj_dir = os.path.dirname(traj_path)
    audit_path = os.path.join(traj_dir, "trajectory_audit.json")
    planner = DPPlanner(env)

    if os.path.exists(traj_path) and (not force_regenerate_trajectories):
        logger.info("发现已有轨迹文件，直接加载: %s", traj_path)
        trajectory_arrays = load_trajectory_arrays(traj_path)
        trajectory_audit = {
            "loaded_existing_trajectories": True,
            "states_shape": tuple(int(v) for v in trajectory_arrays["states"].shape),
            "actions_shape": tuple(int(v) for v in trajectory_arrays["actions"].shape),
            "rewards_shape": tuple(int(v) for v in trajectory_arrays["rewards"].shape),
        }
    else:
        if os.path.exists(traj_path):
            logger.info("按论文对齐版配置重新生成轨迹，并覆盖已有文件: %s", traj_path)
        else:
            logger.info("开始生成论文对齐版 DP 示范轨迹: num_trajectories=%d", config.num_trajectories)

        trajectory_arrays, trajectory_audit = generate_paper_aligned_trajectories(
            planner=planner,
            env=env,
            num_trajectories=config.num_trajectories,
            seed=trajectory_seed,
        )
        save_trajectory_arrays(traj_path, trajectory_arrays)
        save_trajectory_audit(audit_path, trajectory_audit)
        logger.info("论文对齐版轨迹已保存: %s", traj_path)
        logger.info("轨迹审计报告已保存: %s", audit_path)

    dataset = TrajectoryDataset(
        states=trajectory_arrays["states"],
        actions=trajectory_arrays["actions"],
        rewards=trajectory_arrays["rewards"],
    )
    logger.info("Dataset 大小: %d 条轨迹", len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    # ----------------------------------------------------------------
    # Step 3: 初始化 VQ Encoder、Codebook、Decoder
    # ----------------------------------------------------------------
    encoder = VQEncoder(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.lstm_hidden_dim,
        latent_dim=config.latent_dim,
    ).to(device)

    codebook = VQCodebook(
        num_codes=config.num_archetypes,
        code_dim=config.latent_dim,
    ).to(device)

    decoder = VQDecoder(
        state_dim=config.state_dim,
        code_dim=config.latent_dim,
        hidden_dim=config.lstm_hidden_dim,
        action_dim=config.action_dim,
    ).to(device)

    logger.info(
        "模型初始化完成: Encoder params=%d, Codebook params=%d, Decoder params=%d",
        sum(p.numel() for p in encoder.parameters()),
        sum(p.numel() for p in codebook.parameters()),
        sum(p.numel() for p in decoder.parameters()),
    )

    # 优化器：联合训练 encoder + codebook + decoder
    all_params = list(encoder.parameters()) + list(codebook.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(all_params, lr=config.learning_rate)

    # ----------------------------------------------------------------
    # Step 4: 训练循环 — 100 epochs
    # 损失函数 L = L_rec + ||sg[z_e] - z_q||² + β₀ × ||z_e - sg[z_q]||²
    # VQCodebook.quantize() 返回的 commitment_loss = ||sg[z_e] - z_q||²
    # β₀ × ||z_e - sg[z_q]||² 需要在训练循环中额外计算
    #
    # 论文对齐版说明:
    # - 仅保留 Eq. (4) 的三项主损失。
    # - decoder 训练时不喂 previous action；teacher forcing 参数只为兼容接口保留。
    # ----------------------------------------------------------------
    loss_history: List[float] = []

    logger.info("开始训练: %d epochs", config.phase1_epochs)

    for epoch in tqdm(range(1, config.phase1_epochs + 1), desc="Training Epochs"):
        encoder.train()
        codebook.train()
        decoder.train()

        epoch_loss = 0.0
        epoch_rec_loss = 0.0
        epoch_vq_loss = 0.0
        num_batches = 0

        epoch_hist = torch.zeros(config.num_archetypes, dtype=torch.float32)

        for s_demo, a_demo, r_demo in dataloader:
            s_demo = s_demo.to(device)   # (B, h, state_dim)
            a_demo = a_demo.to(device)   # (B, h)
            r_demo = r_demo.to(device)   # (B, h)

            # Phase I, Step 1: Encode
            z_e = encoder(s_demo, a_demo, r_demo)  # (B, latent_dim)

            # Phase I, Step 2: Quantize
            z_q_st, indices, commitment_loss = codebook.quantize(z_e)
            epoch_hist += codebook.usage_histogram(indices).cpu()

            # Phase I, Step 3: Decode
            action_logits = decoder(s_demo, z_q_st, teacher_actions=None)

            # 论文主重建损失 L_rec
            rec_loss = F.cross_entropy(
                action_logits.reshape(-1, action_logits.shape[-1]),
                a_demo.reshape(-1),
                reduction="mean",
            )

            # β₀ × ||z_e - sg[z_q]||²
            z_q_detached = z_q_st.detach()  # sg[z_q] — stop gradient on z_q
            encoder_commitment = config.vq_beta0 * torch.mean((z_e - z_q_detached) ** 2)

            total_loss = rec_loss + commitment_loss + encoder_commitment

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += float(total_loss.item())
            epoch_rec_loss += float(rec_loss.item())
            epoch_vq_loss += float(commitment_loss.item() + encoder_commitment.item())
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_rec = epoch_rec_loss / max(num_batches, 1)
        avg_vq = epoch_vq_loss / max(num_batches, 1)
        loss_history.append(avg_loss)

        epoch_hist_device = epoch_hist.to(device)
        used_code_count = int((epoch_hist.numpy() > 0).sum())
        perplexity = float(codebook.perplexity(epoch_hist_device).item())
        dominant_code_ratio = float(epoch_hist.max().item() / max(epoch_hist.sum().item(), 1.0))

        # 每 10 个 epoch 或首尾 epoch 输出日志
        if epoch == 1 or epoch % 10 == 0 or epoch == config.phase1_epochs:
            logger.info(
                "Epoch %3d/%d — total_loss=%.4f, rec_loss=%.4f, vq_loss=%.4f, "
                "used_code_count=%d, code_perplexity=%.4f, dominant_code_ratio=%.4f",
                epoch,
                config.phase1_epochs,
                avg_loss,
                avg_rec,
                avg_vq,
                used_code_count,
                perplexity,
                dominant_code_ratio,
            )

        # NaN 检测
        if np.isnan(avg_loss):
            logger.error("训练 loss 发散 (NaN)，在 epoch %d 终止训练", epoch)
            break

    # ----------------------------------------------------------------
    # Step 5: 保存模型到 result/phase1_archetype_discovery/
    # ----------------------------------------------------------------
    save_dir = os.path.join(config.result_dir, pair, "phase1_archetype_discovery")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{pair}_vq_model.pt")
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "codebook": codebook.state_dict(),
            "decoder": decoder.state_dict(),
            "loss_history": loss_history,
            "config": {
                "state_dim": config.state_dim,
                "action_dim": config.action_dim,
                "latent_dim": config.latent_dim,
                "num_archetypes": config.num_archetypes,
                "lstm_hidden_dim": config.lstm_hidden_dim,
                "phase1_epochs": config.phase1_epochs,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "vq_beta0": config.vq_beta0,
                "phase1_force_regenerate_trajectories": force_regenerate_trajectories,
                "paper_aligned_phase1": True,
            },
        },
        save_path,
    )
    logger.info("模型已保存到 %s", save_path)

    # 同步把轨迹审计也存到 phase1 结果目录，便于回看训练时的 D 构造质量。
    save_trajectory_audit(os.path.join(save_dir, "trajectory_audit_snapshot.json"), trajectory_audit)

    # ----------------------------------------------------------------
    # Step 6: 输出训练日志摘要
    # ----------------------------------------------------------------
    logger.info("=" * 50)
    logger.info("Phase I 训练完成: pair=%s", pair)
    logger.info("最终 loss: %.4f", loss_history[-1] if loss_history else float("nan"))
    logger.info(
        "最低 loss: %.4f (epoch %d)",
        min(loss_history) if loss_history else float("nan"),
        (loss_history.index(min(loss_history)) + 1) if loss_history else 0,
    )
    logger.info("模型保存路径: %s", save_path)
    logger.info("=" * 50)

    # ----------------------------------------------------------------
    # Step 7: 训练后自动验证
    # ----------------------------------------------------------------
    logger.info("开始执行 Phase I 验证")
    validation_config = ValidationConfig(
        batch_size=getattr(config, "phase1_validation_batch_size", config.batch_size),
        max_eval_samples=getattr(config, "phase1_validation_max_samples", 512),
        replay_num_trajectories=getattr(config, "phase1_replay_num_trajectories", 512),
        dp_oracle_num_horizons=getattr(config, "phase1_dp_oracle_num_horizons", 8),
        dp_bruteforce_horizon=getattr(config, "phase1_dp_bruteforce_horizon", 6),
        replay_abs_tol=validation_replay_abs_tol,
    )
    report = run_phase1_validation(
        encoder=encoder,
        codebook=codebook,
        decoder=decoder,
        planner=planner,
        env=env,
        states_dataframe=train_df,
        prices=prices,
        trajectories=trajectory_arrays,
        config=validation_config,
        device=device,
    )
    validation_path = os.path.join(save_dir, "phase1_validation_report.json")
    save_validation_report(report, validation_path)
    logger.info("Phase I 验证报告已保存到 %s", validation_path)
    logger.info(
        "验证摘要: hard_validation_passed=%s, phase1_ready_for_phase2=%s, "
        "step_accuracy=%.4f, used_code_count=%d, code_perplexity=%.4f, dominant_code_ratio=%.4f",
        report["summary"]["hard_validation_passed"],
        report["summary"]["phase1_ready_for_phase2"],
        report["vq_validation"]["step_accuracy"],
        report["vq_validation"]["used_code_count"],
        report["vq_validation"]["code_perplexity"],
        report["vq_validation"]["dominant_code_ratio"],
    )


def main() -> None:
    config = parse_args()
    train_phase1_with_config(config)


if __name__ == "__main__":
    main()