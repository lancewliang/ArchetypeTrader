"""前置数据处理入口。

读取 ``data/<PAIR>`` 下的 feature split 文件，生成:
    - ``data/<PAIR>/<BATCHID>/horizon_datasets/<split>.npz``
    - ``data/<PAIR>/<BATCHID>/trajectory_datasets/<split>.npz``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_load import DataLoad  # noqa: E402
from src.data.data_preparer import DataPreparer  # noqa: E402
from src.data.resolve_factor import (  # noqa: E402
    FACTORS_ROOT,
    build_feature_columns,
    build_feature_columns_from_file,
    resolve_factor_config_path,
)
from src.store.artifact_store import DataStore  # noqa: E402


DEFAULT_PAIR = "AL"
DEFAULT_BATCHID = "batch_001"
DEFAULT_HORIZON = 72
DEFAULT_FACTOR_SET = "short"
SPLIT_CANDIDATES = {
    "train": ("train.feather", "df_train.feather"),
    "val": (
        "val.feather",
        "df_val.feather",
        "validation.feather",
        "df_validation.feather",
    ),
    "test": ("test.feather", "df_test.feather"),
}


def parse_args() -> argparse.Namespace:
    """解析前置数据处理参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "Generate horizon_datasets and trajectory_datasets under data/<PAIR>."
        )
    )
    parser.add_argument("--pair", default=DEFAULT_PAIR, help="交易标的目录名，默认 AL。")
    parser.add_argument(
        "--batchid",
        default=DEFAULT_BATCHID,
        help="数据准备批次 ID，默认 batch_001。",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
        help="数据根目录，默认项目下的 artifacts。",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="固定 horizon 长度，默认 72。",
    )
    parser.add_argument(
        "--factor-set",
        default=DEFAULT_FACTOR_SET,
        help=(
            "因子配置文件名，不含 .txt 时会自动补齐；默认读取 "
            "src/factors/<PAIR>/short.txt。"
        ),
    )
    parser.add_argument(
        "--factor-file",
        type=Path,
        default=None,
        help="显式因子配置文件路径；提供后优先于 --factor-set。",
    )
    parser.add_argument(
        "--factors-root",
        type=Path,
        default=FACTORS_ROOT,
        help="因子配置根目录，默认项目下的 src/factors。",
    )
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--val-file", type=Path, default=None)
    parser.add_argument("--test-file", type=Path, default=None)
    return parser.parse_args()


def resolve_split_file(
    *,
    pair_dir: Path,
    split_name: str,
    explicit_path: Path | None,
) -> Path | None:
    """返回 split 输入文件路径；找不到时返回 ``None``。"""

    if explicit_path is not None:
        return explicit_path

    for file_name in SPLIT_CANDIDATES[split_name]:
        candidate = pair_dir / file_name
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    """执行前置数据处理主流程。"""

    args = parse_args()
    if args.factor_file is not None:
        feature_columns = build_feature_columns_from_file(args.factor_file)
        factor_source = args.factor_file
    else:
        feature_columns = build_feature_columns(
            pair=args.pair,
            factor_set=args.factor_set,
            factors_root=args.factors_root,
        )
        factor_source = resolve_factor_config_path(
            pair=args.pair,
            factor_set=args.factor_set,
            factors_root=args.factors_root,
        )

    pair_dir = args.data_root / args.pair
    data_store = DataStore(
        pair=args.pair,
        batchid=args.batchid,
        artifacts_root=args.data_root,
    )
    preparer = DataPreparer(
        horizon=args.horizon,
        data_load=DataLoad(feature_columns=feature_columns),
        data_store=data_store,
    )

    split_files = {
        "train": resolve_split_file(
            pair_dir=pair_dir,
            split_name="train",
            explicit_path=args.train_file,
        ),
        "val": resolve_split_file(
            pair_dir=pair_dir,
            split_name="val",
            explicit_path=args.val_file,
        ),
        "test": resolve_split_file(
            pair_dir=pair_dir,
            split_name="test",
            explicit_path=args.test_file,
        ),
    }

    prepared_any = False
    print(
        f"[features] source={factor_source} columns={len(feature_columns)} "
        f"state_features={len(feature_columns) - 1}"
    )
    for split_name, split_file in split_files.items():
        if split_file is None:
            print(f"[skip] {split_name}: feature file not found under {pair_dir}")
            continue

        print(f"[prepare] {split_name}: {split_file}")
        if split_name == "train":
            trajectory_dataset = preparer.prepare_train_dataset(split_file)
        elif split_name == "val":
            trajectory_dataset = preparer.prepare_validation_dataset(split_file)
        else:
            trajectory_dataset = preparer.prepare_test_dataset(split_file)

        horizon_path = data_store.artifact_paths[f"{split_name}_horizon_dataset"]
        trajectory_path = data_store.artifact_paths[f"{split_name}_trajectory_dataset"]
        print(
            f"[done] {split_name}: trajectories={len(trajectory_dataset)} "
            f"horizon={horizon_path} trajectory={trajectory_path}"
        )
        prepared_any = True

    if not prepared_any:
        raise FileNotFoundError(f"no feature split files found under {pair_dir}")


if __name__ == "__main__":
    main()
