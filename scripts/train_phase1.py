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

from src.phase1.phase1_main import Phase1MainConfig, Phase1MainFlow  # noqa: E402
from src.utils import RuntimeUtils  # noqa: E402


PAIR = "FU"
TRAIN_BATCH_ID = "batch_001"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / PAIR / TRAIN_BATCH_ID / "phase1"
SEED = 42
LOG_LEVEL = "INFO"
DETERMINISTIC = True


def initialize_runtime() -> None:
    """初始化入口脚本层面的日志和随机种子。"""

    logger = RuntimeUtils.init_logging(
        name="archetype_trader.phase1",
        log_file=OUTPUT_DIR / "phase1.log",
        level=LOG_LEVEL,
    )
    seed_status = RuntimeUtils.init_random_seed(
        SEED,
        deterministic=DETERMINISTIC,
    )
    logger.info("runtime initialized: seed_status=%s", seed_status)


def create_phase1_flow() -> Phase1MainFlow:
    """使用本脚本硬编码配置创建 Phase I 主流程实例。"""

    config = Phase1MainConfig(
        pair=PAIR,
        train_batch_id=TRAIN_BATCH_ID
    )
    return Phase1MainFlow(config)


def main() -> None:
    """创建并运行 Phase I 主流程。"""

    initialize_runtime()
    flow = create_phase1_flow()
    flow.run()


if __name__ == "__main__":
    main()
