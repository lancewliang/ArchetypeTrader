"""数据准备产出物读写类的接口骨架。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..model.data_types import (
    ArtifactPaths,
    HorizonDataset,
    TrajectoryDataset,
)
from ..data.state_normalizer import StateNormalizer


class DataFileStore:
    """负责计算数据准备产出物路径，并读写中间数据集。

    为什么需要这个类:
        ``horizon_dataset`` 和 ``trajectory_dataset`` 都是需要持久化的产出物。
        将路径规划、保存和读取逻辑从 ``DataPreparer`` 中拆出，可以让
        ``DataPreparer`` 专注于流程编排，避免 I/O 细节散落在训练、测试、
        校验三个入口里。
    """

    def __init__(
        self,
        pair: str | None = None,
        batchid: str | None = None,
        artifacts_root: str | Path = "artifacts",
    ) -> None:
        """初始化数据产物 store。

        参数:
            pair: 可选交易标的，用于子类按标的隔离产物目录。
            batchid: 可选批次 ID，用于子类按批次隔离产物目录。
            artifacts_root: 全阶段产物根目录，默认 ``artifacts``。
        """

        self.pair = pair
        self.batchid = batchid
        self.artifacts_root = Path(artifacts_root)
        self.artifact_paths: ArtifactPaths = {}

    def build_artifact_paths(
        self,
        path: str | Path,
        split_name: str,
    ) -> ArtifactPaths:
        """计算数据准备产出物的存储路径。

        参数:
            path: feature 输入文件路径。
            split_name: 数据集名称，例如 ``train``、``test`` 或 ``validation``。

        输出:
            返回产出物路径字典。
            至少包含:
                ``horizon_dataset``: horizon 中间数据的保存路径。
                ``trajectory_dataset``: demonstration trajectory 数据的保存路径。
                ``state_normalizer``: state 归一化参数的保存路径。

        方法作用:
            根据输入文件和 split 名称，提前规划数据准备阶段的产物位置。

        为什么:
            路径命名规则应集中管理，避免 train/test/validation 各自拼路径，
            导致产物位置和命名不一致。
        """
        root = self._dataset_root(Path(path))
        horizon_path = root / "horizon_datasets" / f"{split_name}.npz"
        trajectory_path = root / "trajectory_datasets" / f"{split_name}.npz"
        normalizer_path = root / "state_normalizer.json"
        feature_normalizers_path = root / "feature_normalizers"
        horizon_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        feature_normalizers_path.mkdir(parents=True, exist_ok=True)

        paths: ArtifactPaths = {
            "horizon_dataset": horizon_path,
            "trajectory_dataset": trajectory_path,
            "state_normalizer": normalizer_path,
            "feature_normalizers": feature_normalizers_path,
            f"{split_name}_horizon_dataset": horizon_path,
            f"{split_name}_trajectory_dataset": trajectory_path,
        }
        self.artifact_paths.update(paths)
        return paths

    def save_horizon_dataset(
        self,
        horizon_dataset: HorizonDataset,
        output_path: str | Path,
    ) -> None:
        """保存 horizon 中间数据。

        参数:
            horizon_dataset: ``HorizonBuilder`` 的输出。新格式为
                ``(states, relative_states, trend_states, prices, depthprices)``。
                ``states`` shape 为 ``[x, h, len(states)]``。
                ``relative_states`` shape 为 ``[x, h, len(relative_states)]``。
                ``trend_states`` shape 为 ``[x, h, len(trend_states)]``。
                ``prices`` shape 为 ``[x, h, 1]``。
                ``depthprices`` shape 为 ``[x, h, 20]``。
            output_path: horizon 中间数据保存路径。

        输出:
            无返回值。

        方法作用:
            将 ``HorizonBuilder`` 生成的 horizon 数据作为独立产物保存。

        为什么:
            horizon 数据是从 feature 文件到 trajectory 数据之间的重要中间结果。
            单独保存可以支持复查 ``close`` 价格切分、状态 shape 和后续 DP 输入。
        """
        states, relative_states, trend_states, prices, depthprices = horizon_dataset
        states = np.asarray(states, dtype=np.float32)
        relative_states = np.asarray(relative_states, dtype=np.float32)
        trend_states = np.asarray(trend_states, dtype=np.float32)
        prices = np.asarray(prices, dtype=np.float32)
        depthprices = np.asarray(depthprices, dtype=np.float32)
        if states.ndim != 3:
            raise ValueError("states must have shape [x, h, feature_dim]")
        if relative_states.ndim != 3:
            raise ValueError(
                "relative_states must have shape [x, h, relative_feature_dim]"
            )
        if trend_states.ndim != 3:
            raise ValueError("trend_states must have shape [x, h, trend_feature_dim]")
        if prices.ndim != 3 or prices.shape[-1] != 1:
            raise ValueError("prices must have shape [x, h, 1]")
        if depthprices.ndim != 3 or depthprices.shape[-1] != 20:
            raise ValueError(
                "depthprices must have shape [x, h, 20] with LOB prices and sizes"
            )
        if (
            relative_states.shape[:2] != states.shape[:2]
            or trend_states.shape[:2] != states.shape[:2]
            or prices.shape[:2] != states.shape[:2]
            or depthprices.shape[:2] != states.shape[:2]
        ):
            raise ValueError("horizon dataset arrays must share [x, h]")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            states=states,
            relative_states=relative_states,
            trend_states=trend_states,
            prices=prices,
            depthprices=depthprices,
        )

    def load_horizon_dataset(
        self,
        split_name: str | Path,
    ) -> HorizonDataset:
        """读取 horizon 中间数据。

        参数:
            split_name: split 名称，例如 ``train``、``val``、``test``。

        输出:
            返回 ``HorizonDataset``，新格式为
            ``(states, relative_states, trend_states, prices, depthprices)``。
            ``states`` shape 为 ``[x, h, len(states)]``。
            ``relative_states`` shape 为 ``[x, h, len(relative_states)]``。
            ``trend_states`` shape 为 ``[x, h, len(trend_states)]``。
            ``prices`` shape 为 ``[x, h, 1]``。
            ``depthprices`` shape 为 ``[x, h, 20]``。

        方法作用:
            从已保存的产出物中恢复 ``HorizonBuilder`` 生成的 horizon 数据。

        为什么:
            数据准备产物需要可复用。后续调试、审计或重新生成
            ``trajectory_dataset`` 时，应能直接读取已固化的 horizon 数据，
            而不是重新读取 feature 文件并重新切分。
        """
        path = self._resolve_dataset_path("horizon", split_name)
        with np.load(path) as payload:
            if "depthprices" not in payload:
                raise ValueError(
                    "horizon dataset is missing depthprices; regenerate it with "
                    "HorizonBuilder before loading"
                )
            depthprices = payload["depthprices"].astype(np.float32, copy=False)
            if depthprices.ndim != 3 or depthprices.shape[-1] != 20:
                raise ValueError(
                    "horizon dataset depthprices must have shape [x, h, 20]; "
                    "regenerate it with LOB prices and sizes"
                )
            states = payload["states"].astype(np.float32, copy=False)
            if "relative_states" in payload:
                relative_states = payload["relative_states"].astype(
                    np.float32,
                    copy=False,
                )
            else:
                relative_states = np.empty((*states.shape[:2], 0), dtype=np.float32)
            if "trend_states" in payload:
                trend_states = payload["trend_states"].astype(np.float32, copy=False)
            else:
                trend_states = np.empty((*states.shape[:2], 0), dtype=np.float32)
            return (
                states,
                relative_states,
                trend_states,
                payload["prices"].astype(np.float32, copy=False),
                depthprices,
            )

    def save_trajectory_dataset(
        self,
        trajectory_dataset: TrajectoryDataset,
        output_path: str | Path,
    ) -> None:
        """保存 demonstration trajectory 数据集。

        参数:
            trajectory_dataset: ``SingleTrade_DP_Planner`` 的输出。
                数据形式为 ``D = [tau_0, tau_1, ..., tau_{n-1}]``。
                每个 ``tau`` 都是
                ``(s_demo, relative_s_demo, trend_s_demo, a_demo, r_demo)``。
            output_path: trajectory 数据集保存路径。

        输出:
            无返回值。

        方法作用:
            将 DP teacher 生成的 demonstration trajectories 保存为训练产物。

        为什么:
            Phase I 训练应消费已经固化的 trajectory 数据集，
            而不是在训练过程中重新读取 feature 文件或重新运行 DP。
        """
        if not trajectory_dataset:
            raise ValueError("trajectory_dataset must not be empty")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        states = np.stack([tau[0] for tau in trajectory_dataset]).astype(
            np.float32,
            copy=False,
        )
        relative_states = np.stack([tau[1] for tau in trajectory_dataset]).astype(
            np.float32,
            copy=False,
        )
        trend_states = np.stack([tau[2] for tau in trajectory_dataset]).astype(
            np.float32,
            copy=False,
        )
        actions = np.stack([tau[3] for tau in trajectory_dataset]).astype(
            np.int64,
            copy=False,
        )
        rewards = np.stack([tau[4] for tau in trajectory_dataset]).astype(
            np.float32,
            copy=False,
        )
        np.savez_compressed(
            output,
            states=states,
            relative_states=relative_states,
            trend_states=trend_states,
            actions=actions,
            rewards=rewards,
        )

    def load_trajectory_dataset(
        self,
        split_name: str | Path,
    ) -> TrajectoryDataset:
        """读取 demonstration trajectory 数据集。

        参数:
            split_name: split 名称，例如 ``train``、``val``、``test``。

        输出:
            返回 ``TrajectoryDataset``。
            数据形式为 ``D = [tau_0, tau_1, ..., tau_{n-1}]``，
            每个 ``tau`` 都是
            ``(s_demo, relative_s_demo, trend_s_demo, a_demo, r_demo)``。

        方法作用:
            从已保存的产出物中恢复 DP teacher 生成的 demonstration trajectories。

        为什么:
            Phase I 训练应直接消费固化后的 ``trajectory_dataset``。
            读取方法可以让训练流程、验证流程和审计流程复用同一份产物，
            避免重复运行 DP teacher。
        """
        path = self._resolve_dataset_path("trajectory", split_name)
        with np.load(path) as payload:
            states = payload["states"].astype(np.float32, copy=False)
            if "relative_states" in payload:
                relative_states = payload["relative_states"].astype(
                    np.float32,
                    copy=False,
                )
            else:
                relative_states = np.empty((*states.shape[:2], 0), dtype=np.float32)
            if "trend_states" in payload:
                trend_states = payload["trend_states"].astype(np.float32, copy=False)
            else:
                trend_states = np.empty((*states.shape[:2], 0), dtype=np.float32)
            actions = payload["actions"].astype(np.int64, copy=False)
            rewards = payload["rewards"].astype(np.float32, copy=False)

        if (
            states.shape[0] != actions.shape[0]
            or states.shape[0] != rewards.shape[0]
            or relative_states.shape[0] != states.shape[0]
            or trend_states.shape[0] != states.shape[0]
        ):
            raise ValueError(f"invalid trajectory dataset file: {path}")
        return [
            (
                states[index],
                relative_states[index],
                trend_states[index],
                actions[index],
                rewards[index],
            )
            for index in range(states.shape[0])
        ]

    def save_feature_normalizers(
        self,
        normalizers: dict[str, StateNormalizer],
        output_dir: str | Path,
    ) -> None:
        """保存 feature block 归一化参数。"""

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for key, normalizer in normalizers.items():
            path = output / f"{key}.json"
            path.write_text(
                json.dumps(normalizer.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        self.artifact_paths["feature_normalizers"] = output

    def load_feature_normalizers(
        self,
        input_dir: str | Path,
    ) -> dict[str, StateNormalizer]:
        """读取 feature block 归一化参数。"""

        directory = Path(input_dir)
        if not directory.exists():
            raise FileNotFoundError(f"feature normalizers not found: {directory}")
        if not directory.is_dir():
            raise ValueError(f"feature normalizers path is not a directory: {directory}")
        normalizers: dict[str, StateNormalizer] = {}
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"invalid feature normalizer payload: {path}")
            normalizers[path.stem] = StateNormalizer.from_dict(payload)
        if not normalizers:
            raise ValueError(f"feature normalizers directory is empty: {directory}")
        return normalizers

    def save_state_normalizer(
        self,
        normalizer: StateNormalizer,
        output_path: str | Path,
    ) -> None:
        """保存 state 归一化参数。"""

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(normalizer.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.artifact_paths["state_normalizer"] = output

    def load_state_normalizer(
        self,
        split_name: str | Path,
    ) -> StateNormalizer:
        """读取 state 归一化参数。"""

        candidate = Path(split_name)
        if candidate.suffix:
            path = candidate
        else:
            stored_path = self.artifact_paths.get("state_normalizer")
            if stored_path is not None:
                path = Path(stored_path)
            else:
                path = self._dataset_root() / "state_normalizer.json"
        if not path.exists():
            raise FileNotFoundError(f"state normalizer not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"invalid state normalizer payload: {path}")
        return StateNormalizer.from_dict(payload)

    def _dataset_root(self, input_path: Path | None = None) -> Path:
        """返回数据集产物根目录。"""

        root = self.artifacts_root
        if self.pair:
            root = root / self.pair
            if self.batchid:
                root = root / self.batchid
            return root
        if input_path is not None:
            return input_path.parent
        return root

    def _resolve_dataset_path(
        self,
        dataset_kind: str,
        split_name: str | Path,
    ) -> Path:
        """解析 horizon/trajectory 数据集的读取路径。"""

        candidate = Path(split_name)
        if candidate.suffix:
            return candidate

        split = str(split_name)
        key = f"{split}_{dataset_kind}_dataset"
        stored_path = self.artifact_paths.get(key)
        if stored_path is not None:
            return Path(stored_path)

        fallback_key = f"{dataset_kind}_dataset"
        fallback_path = self.artifact_paths.get(fallback_key)
        if fallback_path is not None and Path(fallback_path).stem == split:
            return Path(fallback_path)

        directory = (
            "horizon_datasets"
            if dataset_kind == "horizon"
            else "trajectory_datasets"
        )
        return self._dataset_root() / directory / f"{split}.npz"


DataStore = DataFileStore
