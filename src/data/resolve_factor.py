"""因子配置解析和特征列清单构建工具。"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path


FIXED_FEATURES = [
    "close",
    "ask1_price",
    "ask1_size",
    "bid1_price",
    "bid1_size",
    "ask2_price",
    "ask2_size",
    "bid2_price",
    "bid2_size",
    "ask3_price",
    "ask3_size",
    "bid3_price",
    "bid3_size",
    "ask4_price",
    "ask4_size",
    "bid4_price",
    "bid4_size",
    "ask5_price",
    "ask5_size",
    "bid5_price",
    "bid5_size",
    "total_trade_volume",
    "turnover",
    "open_interest",
]
FACTORS_ROOT = Path(__file__).resolve().parents[1] / "factors"


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def read_factor_config(config_path: str | Path) -> list[str]:
    """读取单个因子配置文件。

    配置文件支持每行一个因子名，也支持用逗号或空白分隔；空行和 ``#`` 注释会被忽略。
    """

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"factor config file not found: {path}")
    if not path.is_file():
        raise ValueError(f"factor config path is not a file: {path}")

    factors: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        factors.extend(token for token in line.replace(",", " ").split() if token)
    if not factors:
        raise ValueError(f"factor config file is empty: {path}")
    return _dedupe_preserve_order(factors)


def resolve_factor_config_path(
    *,
    pair: str,
    factor_set: str,
    factors_root: str | Path = FACTORS_ROOT,
) -> Path:
    """根据交易标的和因子集合名解析 ``src/factors/<pair>/<factor_set>.txt``。"""

    file_name = factor_set if factor_set.endswith(".txt") else f"{factor_set}.txt"
    return Path(factors_root) / pair / file_name


def build_feature_columns(
    *,
    pair: str,
    factor_set: str,
    factors_root: str | Path = FACTORS_ROOT,
    fixed_features: Sequence[str] = FIXED_FEATURES,
) -> list[str]:
    """返回固定行情列和配置因子的合并列清单。"""

    config_path = resolve_factor_config_path(
        pair=pair,
        factor_set=factor_set,
        factors_root=factors_root,
    )
    return build_feature_columns_from_file(
        config_path,
        fixed_features=fixed_features,
    )


def build_feature_columns_from_file(
    config_path: str | Path,
    *,
    fixed_features: Sequence[str] = FIXED_FEATURES,
) -> list[str]:
    """从显式配置文件返回固定行情列和配置因子的合并列清单。"""

    configured_factors = read_factor_config(config_path)
    return _dedupe_preserve_order([*fixed_features, *configured_factors])
