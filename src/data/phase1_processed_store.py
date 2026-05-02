"""Phase I processed data manifest and split artifact store.

This store is the boundary between offline Phase I data processing and model
training. It persists sampled horizons separately from DP teacher labels, then
loads and validates the pair by ``sample_id`` for manifest-mode training.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.data.horizon_builder import HorizonRecord
from src.data.schema import InputSchema
from src.planners.demo_generator import RejectStats
from src.trading.cost_model import ExecutionBook
from src.utils import feather_io
from src.utils.feather_io import atomic_write_json, read_json


class Phase1ProcessedStoreError(ValueError):
    """Raised when processed Phase I artifacts fail validation."""


@dataclass(frozen=True)
class Phase1SplitArtifact:
    split: str
    window_index_path: str
    sampled_horizons_path: str
    dp_teacher_path: str
    num_horizons: int
    reject_stats_path: str = ""

    @classmethod
    def from_dict(cls, split: str, payload: Mapping[str, Any]) -> "Phase1SplitArtifact":
        return cls(
            split=split,
            window_index_path=str(payload.get("window_index_path", "")),
            sampled_horizons_path=str(payload.get("sampled_horizons_path", "")),
            dp_teacher_path=str(payload.get("dp_teacher_path", "")),
            num_horizons=int(payload.get("num_horizons", 0)),
            reject_stats_path=str(payload.get("reject_stats_path", "")),
        )

    def to_dict(self) -> dict:
        payload = {
            "window_index_path": self.window_index_path,
            "sampled_horizons_path": self.sampled_horizons_path,
            "dp_teacher_path": self.dp_teacher_path,
            "num_horizons": self.num_horizons,
        }
        if self.reject_stats_path:
            payload["reject_stats_path"] = self.reject_stats_path
        return payload


@dataclass(frozen=True)
class Phase1DataProcessManifest:
    version: int
    phase: str
    pair: str
    data_batch_id: str
    artifact_dir: str
    created_at: str
    input_files: Dict[str, str]
    input_schema_path: str
    schema_hash: str
    data_process_hash: str
    dp_teacher_hash: str
    feature_source: Dict[str, Any]
    splits: Dict[str, Phase1SplitArtifact]
    manifest_path: Optional[Path] = None
    input_file_audit: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, manifest_path: Optional[Path] = None
    ) -> "Phase1DataProcessManifest":
        splits = {
            split: Phase1SplitArtifact.from_dict(split, split_payload)
            for split, split_payload in dict(payload.get("splits", {})).items()
        }
        return cls(
            version=int(payload.get("version", 0)),
            phase=str(payload.get("phase", "")),
            pair=str(payload.get("pair", "")),
            data_batch_id=str(payload.get("data_batch_id", "")),
            artifact_dir=str(payload.get("artifact_dir", "")),
            created_at=str(payload.get("created_at", "")),
            input_files=dict(payload.get("input_files", {})),
            input_schema_path=str(payload.get("input_schema_path", "")),
            schema_hash=str(payload.get("schema_hash", "")),
            data_process_hash=str(payload.get("data_process_hash", "")),
            dp_teacher_hash=str(payload.get("dp_teacher_hash", "")),
            feature_source=dict(payload.get("feature_source", {})),
            splits=splits,
            manifest_path=manifest_path,
            input_file_audit=payload.get("input_file_audit"),
        )

    @classmethod
    def load(cls, path: Path | str) -> "Phase1DataProcessManifest":
        target = Path(path)
        return cls.from_dict(read_json(target), manifest_path=target)

    def to_dict(self) -> dict:
        payload = {
            "version": self.version,
            "phase": self.phase,
            "pair": self.pair,
            "data_batch_id": self.data_batch_id,
            "artifact_dir": self.artifact_dir,
            "created_at": self.created_at,
            "input_files": self.input_files,
            "input_schema_path": self.input_schema_path,
            "schema_hash": self.schema_hash,
            "data_process_hash": self.data_process_hash,
            "dp_teacher_hash": self.dp_teacher_hash,
            "feature_source": self.feature_source,
            "splits": {k: v.to_dict() for k, v in self.splits.items()},
        }
        if self.input_file_audit is not None:
            payload["input_file_audit"] = self.input_file_audit
        return payload

    @property
    def base_dir(self) -> Path:
        if self.manifest_path is not None:
            return self.manifest_path.parent
        return Path(self.artifact_dir)

    def resolve(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return self.base_dir / candidate


class Phase1ProcessedStore:
    """Persist and validate Phase I sampled horizons + DP teacher artifacts."""

    def __init__(self, artifact_dir: Path | str) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Save ----------

    def save_sampled_horizons(
        self,
        split: str,
        records: Sequence[HorizonRecord],
        *,
        schema_hash: str,
        data_process_hash: str,
    ) -> Path:
        import polars as pl

        rows = []
        for rec in records:
            rows.append(
                {
                    "sample_id": rec.sample_id,
                    "pair": rec.pair,
                    "split": rec.split,
                    "start_index": rec.start_index,
                    "end_index": rec.end_index,
                    "last_execution_row": rec.last_execution_row,
                    "last_markout_row": rec.last_markout_row,
                    "strata_label": rec.strata_label,
                    "states": rec.states,
                    "prices": rec.prices,
                    "execution_books": json.dumps(
                        [_execution_book_to_dict(book) for book in rec.execution_books],
                        separators=(",", ":"),
                    ),
                    "is_augmented": rec.is_augmented,
                    "augmentation_type": rec.augmentation_type,
                    "_schema_hash": schema_hash,
                    "_data_process_hash": data_process_hash,
                }
            )
        return feather_io.write_ipc(
            pl.DataFrame(rows), self.artifact_dir / f"sampled_horizons_{split}.feather"
        )

    def save_dp_teacher(
        self,
        split: str,
        records: Sequence[HorizonRecord],
        reject_stats: RejectStats,
        *,
        schema_hash: str,
        data_process_hash: str,
        dp_teacher_hash: str,
    ) -> Path:
        import polars as pl

        rows = []
        counts = list(reject_stats.per_horizon_reject_count or [])
        rates = list(reject_stats.per_horizon_reject_rate or [])
        for idx, rec in enumerate(records):
            actions = list(rec.actions or [])
            rewards = list(rec.rewards or [])
            switch_seq = actions[:-1] if len(actions) > 1 else actions
            rows.append(
                {
                    "sample_id": rec.sample_id,
                    "pair": rec.pair,
                    "split": rec.split,
                    "actions": actions,
                    "rewards": rewards,
                    "teacher_return": float(sum(rewards)),
                    "num_switches": int(
                        sum(
                            1
                            for pos in range(1, len(switch_seq))
                            if switch_seq[pos] != switch_seq[pos - 1]
                        )
                    ),
                    "is_no_trade": bool(actions and all(a == 1 for a in actions)),
                    "reject_transition_count": int(counts[idx]) if idx < len(counts) else 0,
                    "reject_transition_rate": float(rates[idx]) if idx < len(rates) else 0.0,
                    "_schema_hash": schema_hash,
                    "_data_process_hash": data_process_hash,
                    "_dp_teacher_hash": dp_teacher_hash,
                }
            )
        return feather_io.write_ipc(
            pl.DataFrame(rows), self.artifact_dir / f"dp_teacher_{split}.feather"
        )

    def save_reject_stats(self, split: str, reject_stats: RejectStats) -> Path:
        payload = {
            "dataset_reject_rate": float(reject_stats.dataset_reject_rate),
            "per_horizon_reject_count": list(reject_stats.per_horizon_reject_count),
            "per_horizon_reject_rate": list(reject_stats.per_horizon_reject_rate),
            "worst_reject_horizons": list(reject_stats.worst_reject_horizons),
            "reject_by_action_pair": dict(reject_stats.reject_by_action_pair),
        }
        return atomic_write_json(payload, self.artifact_dir / f"reject_stats_{split}.json")

    def write_manifest(self, payload: Mapping[str, Any]) -> Path:
        return atomic_write_json(dict(payload), self.artifact_dir / "data_process_manifest.json")

    # ---------- Load ----------

    def load_manifest(self, manifest: Path | str | Phase1DataProcessManifest) -> Phase1DataProcessManifest:
        if isinstance(manifest, Phase1DataProcessManifest):
            return manifest
        loaded = Phase1DataProcessManifest.load(Path(manifest))
        self.validate_manifest(loaded)
        return loaded

    def validate_manifest(self, manifest: Phase1DataProcessManifest) -> None:
        if manifest.version != 1:
            raise Phase1ProcessedStoreError(f"unsupported manifest version: {manifest.version}")
        if manifest.phase != "phase1_data_process":
            raise Phase1ProcessedStoreError(f"unexpected manifest phase: {manifest.phase}")
        missing = [split for split in ("train", "val", "test") if split not in manifest.splits]
        if missing:
            raise Phase1ProcessedStoreError(f"manifest missing splits: {missing}")
        for split, artifact in manifest.splits.items():
            for attr in ("sampled_horizons_path", "dp_teacher_path"):
                path = manifest.resolve(getattr(artifact, attr))
                if not path.exists():
                    raise FileNotFoundError(f"processed artifact missing: {path}")

    def load_schema(self, manifest: Path | str | Phase1DataProcessManifest) -> InputSchema:
        manifest_obj = self.load_manifest(manifest)
        schema_path = manifest_obj.resolve(manifest_obj.input_schema_path)
        payload = read_json(schema_path)
        schema = InputSchema(**payload)
        actual_hash = stable_hash(schema.to_dict())
        if actual_hash != manifest_obj.schema_hash:
            raise Phase1ProcessedStoreError(
                f"input_schema hash mismatch: manifest={manifest_obj.schema_hash} actual={actual_hash}"
            )
        return schema

    def load_reject_stats(
        self, manifest: Path | str | Phase1DataProcessManifest, split: str
    ) -> RejectStats:
        manifest_obj = self.load_manifest(manifest)
        artifact = manifest_obj.splits[split]
        if artifact.reject_stats_path:
            payload = read_json(manifest_obj.resolve(artifact.reject_stats_path))
            return RejectStats(
                dataset_reject_rate=float(payload.get("dataset_reject_rate", 0.0)),
                per_horizon_reject_count=list(payload.get("per_horizon_reject_count", [])),
                per_horizon_reject_rate=list(payload.get("per_horizon_reject_rate", [])),
                worst_reject_horizons=list(payload.get("worst_reject_horizons", [])),
                reject_by_action_pair=dict(payload.get("reject_by_action_pair", {})),
            )
        teacher = feather_io.read_ipc(manifest_obj.resolve(artifact.dp_teacher_path))
        rates = [float(v) for v in teacher.get_column("reject_transition_rate").to_list()]
        counts = [int(v) for v in teacher.get_column("reject_transition_count").to_list()]
        return RejectStats(
            dataset_reject_rate=float(sum(rates) / max(len(rates), 1)),
            per_horizon_reject_count=counts,
            per_horizon_reject_rate=rates,
        )

    def load_records(
        self, manifest: Path | str | Phase1DataProcessManifest, split: str
    ) -> List[HorizonRecord]:
        manifest_obj = self.load_manifest(manifest)
        if split not in manifest_obj.splits:
            raise Phase1ProcessedStoreError(f"manifest does not contain split={split}")
        artifact = manifest_obj.splits[split]
        sampled = self._load_sampled_horizons(manifest_obj, split, artifact)
        teacher_rows = self._load_teacher_rows(manifest_obj, split, artifact)
        records = self.join_horizons_with_teacher(sampled, teacher_rows)
        if len(records) != artifact.num_horizons:
            raise Phase1ProcessedStoreError(
                f"{split} num_horizons mismatch: manifest={artifact.num_horizons} actual={len(records)}"
            )
        return records

    def join_horizons_with_teacher(
        self,
        sampled: Sequence[HorizonRecord],
        teacher_rows: Mapping[str, Mapping[str, Any]],
    ) -> List[HorizonRecord]:
        sampled_ids = [rec.sample_id for rec in sampled]
        if len(sampled_ids) != len(set(sampled_ids)):
            raise Phase1ProcessedStoreError("duplicate sample_id in sampled horizons")
        teacher_ids = list(teacher_rows.keys())
        if len(teacher_ids) != len(set(teacher_ids)):
            raise Phase1ProcessedStoreError("duplicate sample_id in dp teacher")
        sampled_set = set(sampled_ids)
        teacher_set = set(teacher_ids)
        missing = sampled_set - teacher_set
        extra = teacher_set - sampled_set
        if missing or extra:
            raise Phase1ProcessedStoreError(
                f"sample_id mismatch: missing_teacher={sorted(missing)} extra_teacher={sorted(extra)}"
            )

        out: List[HorizonRecord] = []
        for rec in sampled:
            row = teacher_rows[rec.sample_id]
            actions = list(row["actions"] or [])
            rewards = list(row["rewards"] or [])
            horizon_len = len(rec.states)
            if len(actions) != horizon_len or len(rewards) != horizon_len:
                raise Phase1ProcessedStoreError(
                    f"{rec.sample_id} action/reward length mismatch: "
                    f"horizon={horizon_len} actions={len(actions)} rewards={len(rewards)}"
                )
            rec.actions = actions
            rec.rewards = rewards
            out.append(rec)
        return out

    def _load_sampled_horizons(
        self,
        manifest: Phase1DataProcessManifest,
        split: str,
        artifact: Phase1SplitArtifact,
    ) -> List[HorizonRecord]:
        path = manifest.resolve(artifact.sampled_horizons_path)
        frame = feather_io.read_ipc(path)
        self._validate_unique(frame, "sampled_horizons", path)
        self._validate_column_value(frame, "pair", manifest.pair, path)
        self._validate_column_value(frame, "split", split, path)
        self._validate_column_value(frame, "_schema_hash", manifest.schema_hash, path)
        self._validate_column_value(frame, "_data_process_hash", manifest.data_process_hash, path)

        records: List[HorizonRecord] = []
        for row in frame.iter_rows(named=True):
            records.append(
                HorizonRecord(
                    sample_id=row["sample_id"],
                    start_index=int(row["start_index"]),
                    end_index=int(row["end_index"]),
                    last_execution_row=row.get("last_execution_row"),
                    last_markout_row=row.get("last_markout_row"),
                    pair=row["pair"],
                    split=row["split"],
                    strata_label=row["strata_label"],
                    states=row["states"],
                    prices=row["prices"],
                    execution_books=[
                        _execution_book_from_dict(item)
                        for item in json.loads(row.get("execution_books") or "[]")
                    ],
                    is_augmented=bool(row["is_augmented"]),
                    augmentation_type=row["augmentation_type"],
                )
            )
        return records

    def _load_teacher_rows(
        self,
        manifest: Phase1DataProcessManifest,
        split: str,
        artifact: Phase1SplitArtifact,
    ) -> Dict[str, Mapping[str, Any]]:
        path = manifest.resolve(artifact.dp_teacher_path)
        frame = feather_io.read_ipc(path)
        self._validate_unique(frame, "dp_teacher", path)
        self._validate_column_value(frame, "pair", manifest.pair, path)
        self._validate_column_value(frame, "split", split, path)
        self._validate_column_value(frame, "_schema_hash", manifest.schema_hash, path)
        self._validate_column_value(frame, "_data_process_hash", manifest.data_process_hash, path)
        self._validate_column_value(frame, "_dp_teacher_hash", manifest.dp_teacher_hash, path)
        return {row["sample_id"]: row for row in frame.iter_rows(named=True)}

    @staticmethod
    def _validate_unique(frame, label: str, path: Path) -> None:
        if frame.height == 0:
            return
        ids = frame.get_column("sample_id").to_list()
        if len(ids) != len(set(ids)):
            raise Phase1ProcessedStoreError(f"duplicate sample_id in {label}: {path}")

    @staticmethod
    def _validate_column_value(frame, column: str, expected: str, path: Path) -> None:
        if column not in frame.columns:
            raise Phase1ProcessedStoreError(f"{path} missing required column {column}")
        values = set(frame.get_column(column).to_list())
        if values != {expected}:
            raise Phase1ProcessedStoreError(
                f"{path} column {column} mismatch: expected={expected} values={sorted(values)}"
            )


def stable_hash(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _execution_book_to_dict(book: ExecutionBook) -> dict:
    return {
        "ask_prices": list(book.ask_prices),
        "ask_sizes": list(book.ask_sizes),
        "bid_prices": list(book.bid_prices),
        "bid_sizes": list(book.bid_sizes),
        "mark_price": float(book.mark_price),
    }


def _execution_book_from_dict(payload: Mapping[str, Any]) -> ExecutionBook:
    return ExecutionBook(
        ask_prices=tuple(float(v) for v in payload.get("ask_prices", [])),
        ask_sizes=tuple(float(v) for v in payload.get("ask_sizes", [])),
        bid_prices=tuple(float(v) for v in payload.get("bid_prices", [])),
        bid_sizes=tuple(float(v) for v in payload.get("bid_sizes", [])),
        mark_price=float(payload.get("mark_price", 0.0)),
    )
