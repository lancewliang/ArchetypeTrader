"""Phase I 训练入口。

当前入口不接收任何 CLI 参数，所有运行参数先在本文件中硬编码。
后续需要恢复命令行参数时，只在这个脚本中解析并组装 ``Phase1MainConfig``。
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train.phase1_main import Phase1MainConfig, Phase1MainFlow  # noqa: E402


PAIR = "AL"
TRAIN_BATCH_ID = "batch_001"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / PAIR / TRAIN_BATCH_ID / "phase1"

DATA_PROCESS_MANIFEST = None
TRAIN_FILE = PROJECT_ROOT / "data" / PAIR / "train.feather"
VAL_FILE = PROJECT_ROOT / "data" / PAIR / "val.feather"
TEST_FILE = PROJECT_ROOT / "data" / PAIR / "test.feather"

EPOCHS = 100
PRETRAIN_EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
DEVICE = "cpu"
SEED = 42


def create_phase1_flow() -> Phase1MainFlow:
    """使用本脚本硬编码配置创建 Phase I 主流程实例。"""

    config = Phase1MainConfig(
        pair=PAIR,
        train_batch_id=TRAIN_BATCH_ID,
        output_dir=OUTPUT_DIR,
        data_process_manifest=DATA_PROCESS_MANIFEST,
        train_file=TRAIN_FILE,
        val_file=VAL_FILE,
        test_file=TEST_FILE,
        epochs=EPOCHS,
        pretrain_epochs=PRETRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        seed=SEED,
    )
    return Phase1MainFlow(config)


def main() -> None:
    """创建并运行 Phase I 主流程。"""

    flow = create_phase1_flow()
    flow.run()


if __name__ == "__main__":
    main()
