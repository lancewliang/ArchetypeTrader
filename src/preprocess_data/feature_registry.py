"""标的级因子清单加载与固定字段组合。

设计文档锚点: ``docs/design/20260502_phase1_factor_change_log_design.md``。

本模块只负责把固定字段和 ``src/factors/{PAIR}/{profile}.txt`` 合成为
稳定的 ``feature_columns``，不读取市场数据、不做特征工程。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


FIXED_FEATURES = [
    "close",
    "ask1_price", "ask1_size", "bid1_price", "bid1_size",
    "ask2_price", "ask2_size", "bid2_price", "bid2_size",
    "ask3_price", "ask3_size", "bid3_price", "bid3_size",
    "ask4_price", "ask4_size", "bid4_price", "bid4_size",
    "ask5_price", "ask5_size", "bid5_price", "bid5_size",
    "total_trade_volume", "turnover", "open_interest",
]


@dataclass(frozen=True)
class FeatureSelectionSpec:
    """固定字段 + 标的级因子文件的解析结果。"""

    pair: str
    profile: str
    factor_list_path: str
    fixed_features: List[str]
    configured_factors: List[str]
    feature_columns: List[str]
    price_column: str = "close"
    deduplicated_features: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["mode"] = "fixed_plus_factor_list"
        return payload


def default_factor_list_path(pair: str, profile: str) -> Path:
    """返回仓库内默认因子清单路径。"""

    root = Path(__file__).resolve().parents[2]
    return root / "src" / "factors" / pair / f"{profile}.txt"


def resolve_factor_list_path(
    pair: str,
    profile: str,
    factor_list_file: Optional[str] = None,
) -> Path:
    """解析显式路径或默认 ``src/factors/{PAIR}/{profile}.txt``。"""

    if not factor_list_file:
        return default_factor_list_path(pair, profile)
    candidate = Path(factor_list_file)
    if candidate.is_absolute():
        return candidate
    root = Path(__file__).resolve().parents[2]
    repo_relative = root / candidate
    if repo_relative.exists():
        return repo_relative
    return candidate


def load_feature_selection(
    pair: str,
    profile: str = "short",
    factor_list_file: Optional[str] = None,
) -> FeatureSelectionSpec:
    """加载固定字段和标的级因子清单。

    Raises
    ------
    FileNotFoundError : 因子文件不存在。
    ValueError : 因子文件包含 ``close`` 或空字段。
    """

    path = resolve_factor_list_path(pair, profile, factor_list_file)
    if not path.exists():
        raise FileNotFoundError(
            f"因子清单不存在: pair={pair!r}, profile={profile!r}, path={path}"
        )

    configured = _read_factor_lines(path)
    if "close" in configured:
        raise ValueError("因子清单不得包含 close；close 只能作为 price_column 使用")

    seen = set()
    feature_columns: List[str] = []
    deduplicated: List[str] = []
    for col in FIXED_FEATURES:
        if col == "close":
            continue
        if col not in seen:
            feature_columns.append(col)
            seen.add(col)
    for col in configured:
        if col in seen:
            deduplicated.append(col)
            continue
        feature_columns.append(col)
        seen.add(col)

    return FeatureSelectionSpec(
        pair=pair,
        profile=profile,
        factor_list_path=str(path),
        fixed_features=list(FIXED_FEATURES),
        configured_factors=configured,
        feature_columns=feature_columns,
        deduplicated_features=deduplicated,
    )


def _read_factor_lines(path: Path) -> List[str]:
    factors: List[str] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if (line.startswith('"') and line.endswith('"')) or (
            line.startswith("'") and line.endswith("'")
        ):
            line = line[1:-1].strip()
        if not line:
            raise ValueError(f"因子清单 {path} 第 {line_no} 行为空字段")
        factors.append(line)
    return factors
