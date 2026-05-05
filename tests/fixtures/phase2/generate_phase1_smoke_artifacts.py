"""生成 Phase I smoke 产物，供 Phase II 测试使用。

设计文档锚点: Phase II 执行计划 §Step 10。

最小输出目录:
artifacts/TEST/smoke_phase1/phase1/
  decoder.pt
  codebook.pt
  sampled_horizon_labels_train.feather
  sampled_horizon_labels_val.feather
  sampled_horizon_labels_test.feather
  input_schema.json
  reward_normalizer.json
  feature_provenance.json
  phase1_config.yaml
  phase1_report.json
  checkpoint_manifest.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def generate_smoke_phase1_artifacts(output_dir: Path) -> None:
    """生成最小可用的 Phase I 冻结产物。"""
    import numpy as np
    import polars as pl
    import torch
    import yaml

    from src.models.vq_archetype import ArchetypeDecoder

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_dim = 3  # feature_return_1, feature_vol_4, feature_momentum_8
    code_dim = 16
    hidden_dim = 32
    num_codes = 10
    horizon = 8

    # 1. decoder.pt
    decoder = ArchetypeDecoder(feature_dim, code_dim, hidden_dim)
    torch.save(decoder.state_dict(), output_dir / "decoder.pt")

    # 2. codebook.pt
    codebook = torch.randn(num_codes, code_dim) * 0.1
    torch.save(codebook, output_dir / "codebook.pt")

    # 3. encoder.pt (可选)
    # 跳过

    # 4. horizon labels
    for split, n in [("train", 12), ("val", 6), ("test", 6)]:
        df = pl.DataFrame({
            "sample_id": [f"p1_{split}_{i:04d}" for i in range(n)],
            "start_index": [i * horizon for i in range(n)],
            "code_label": [i % num_codes for i in range(n)],
        })
        df.write_ipc(output_dir / f"sampled_horizon_labels_{split}.feather")

    # 5. input_schema.json
    schema = {
        "timestamp_column": "timestamp",
        "price_column": "close",
        "feature_columns": ["feature_return_1", "feature_vol_4", "feature_momentum_8"],
        "excluded_columns": ["timestamp", "close"],
        "orderbook_columns": [
            f"{side}{i}_{field}"
            for side in ["ask", "bid"]
            for i in range(1, 6)
            for field in ["price", "size"]
        ],
        "num_rows": 96,
    }
    with open(output_dir / "input_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    # 6. reward_normalizer.json
    norm = {
        "method": "train_reward_robust",
        "center": 0.0,
        "scale": 1.0,
        "clip_value": 8.0,
        "kurtosis": 3.0,
        "skew": 0.0,
    }
    with open(output_dir / "reward_normalizer.json", "w") as f:
        json.dump(norm, f, indent=2)

    # 7. feature_provenance.json
    provenance = {
        "feature_columns": {
            "feature_return_1": {
                "source_columns": ["close"],
                "lookback_start_bars": -1,
                "lookback_end_bars": 0,
                "publish_delay_bars": 0,
                "fit_scope": "train_only",
                "uses_future_rows": False,
            },
            "feature_vol_4": {
                "source_columns": ["close"],
                "lookback_start_bars": -4,
                "lookback_end_bars": 0,
                "publish_delay_bars": 0,
                "fit_scope": "train_only",
                "uses_future_rows": False,
            },
            "feature_momentum_8": {
                "source_columns": ["close"],
                "lookback_start_bars": -8,
                "lookback_end_bars": 0,
                "publish_delay_bars": 0,
                "fit_scope": "train_only",
                "uses_future_rows": False,
            },
        },
    }
    with open(output_dir / "feature_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    # 8. phase1_config.yaml
    p1_config = {
        "pair": "TEST",
        "train_batch_id": "smoke_phase1",
        "horizon": horizon,
        "dp": {
            "cost_config": {
                "reward_alignment": "paper_formula",
                "commission_rate": 0.0002,
                "slippage_model": "lob_depth",
                "book_levels": 5,
                "mark_price": "mid_price",
                "execution_lag": 0,
                "insufficient_depth_policy": "reject_transition",
            },
            "max_position": 1,
        },
        "model": {
            "hidden_dim": hidden_dim,
            "code_dim": code_dim,
            "num_codes": num_codes,
        },
    }
    with open(output_dir / "phase1_config.yaml", "w") as f:
        yaml.safe_dump(p1_config, f)

    # 9. phase1_report.json
    report = {
        "fatal_collapse": False,
        "code_assignment_drift_warning": False,
        "hindsight_bias_warning": "ok",
        "config_hash": "smoke_test_hash",
        "code_usage": {"counts": [10] * num_codes},
        "reconstruction_accuracy": 0.8,
        "weighted_reconstruction_accuracy": 0.75,
        "non_flat_accuracy": 0.7,
        "single_trade_consistency_rate": 0.9,
        "no_trade_ratio": 0.1,
        "reward_alignment": "paper_formula",
        "reward_normalization_resolved": "train_reward_robust",
        "reward_norm_clip_ratio": 0.01,
        "dataset_reject_rate": 0.0,
        "stratification_mode": "hindsight_horizon",
        "is_hindsight_stratification": True,
        "prospective_diagnostic_required": False,
        "diagnostic_pair_batch_id": None,
        "phase1_composite_score": 0.7,
        "best_epoch": 10,
        "best_checkpoint_path": str(output_dir / "best_vq_model.pt"),
        "selection_metric": "phase1_composite_score",
        "composite_score_sensitivity": {},
    }
    with open(output_dir / "phase1_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # 10. checkpoint_manifest.json
    manifest = [{
        "epoch": 10,
        "path": str(output_dir / "best_vq_model.pt"),
        "file_hash": "smoke_hash",
        "metrics_path": "",
        "verdict": "promote_to_best",
        "reasons": [],
        "composite_score": 0.7,
        "is_best": True,
        "is_periodic": False,
    }]
    with open(output_dir / "checkpoint_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Phase I smoke artifacts 已生成到 {output_dir}")


if __name__ == "__main__":
    output_dir = Path("artifacts/TEST/smoke_phase1/phase1")
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_smoke_phase1_artifacts(output_dir)
