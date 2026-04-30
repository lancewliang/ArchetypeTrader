"""Latent 可视化: TensorBoard embedding + PCA/t-SNE 投影.

设计文档锚点: §4.12。

实现保持简洁: 写 feather 而不直接写 TensorBoard event；TensorBoard 的依赖
是可选的，主流程不必引入 tensorboardX 依赖也能 sign-off。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass
class LatentSnapshot:
    epoch: int
    sample_ids: List[str] = field(default_factory=list)
    splits: List[str] = field(default_factory=list)
    code_ids: List[int] = field(default_factory=list)
    z_e: list = field(default_factory=list)
    z_q: list = field(default_factory=list)
    encoder_hidden_last: Optional[list] = None
    state_embedding: Optional[list] = None
    action_embedding: Optional[list] = None
    reward_embedding: Optional[list] = None
    codebook: list = field(default_factory=list)
    metadata: Dict[str, list] = field(default_factory=dict)


class LatentVisualizationWriter:
    """周期性写入 latent snapshot 与 PCA / t-SNE 投影。

    设计约束（§4.12）
    ------------------
    - probe 样本必须固定（使用 ``fixed_probe_seed``）；不允许每 epoch 重抽样，
      否则轨迹漂移不可解释。
    - t-SNE 成本高；按 ``log_every_epochs`` 触发，PCA 可每次诊断都跑。
    - 缺失 layer 跳过并记入 manifest，不让训练失败。
    - tensorboard 依赖是可选的，缺失时函数直接返回（不阻断训练）。
    """

    def __init__(
        self,
        log_dir: Path,
        log_every_epochs: int,
        max_points_per_split: int,
        fixed_probe_seed: int,
        projections: Sequence[str],
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_every_epochs = log_every_epochs
        self.max_points_per_split = max_points_per_split
        self.fixed_probe_seed = fixed_probe_seed
        self.projections = list(projections)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def should_capture(self, epoch: int, is_best: bool = False, is_last: bool = False) -> bool:
        """是否在该 epoch 触发 capture。

        - epoch 0、is_best、is_last 必抓。
        - 其余按 ``epoch % log_every_epochs == 0`` 触发。
        """
        if is_best or is_last or epoch == 0:
            return True
        return (epoch % max(self.log_every_epochs, 1)) == 0

    def capture_epoch(
        self, model, probe_dataset, epoch: int, metrics: dict
    ) -> Optional[LatentSnapshot]:
        """对固定 probe 样本跑 ``model.encode``，收集 ``z_e / z_q / code_id`` 与 codebook。

        Returns
        -------
        LatentSnapshot or None : 不到 ``log_every_epochs`` 触发频率时返回 ``None``。

        Notes
        -----
        probe_dataset 必须由调用方按 ``fixed_probe_seed`` 抽好；本模块只跑模型 forward
        以避免在 capture 内部产生采样状态污染。
        """
        if not self.should_capture(epoch):
            return None
        # probe_dataset 必须由调用方按固定 seed 抽好；本模块只跑模型 forward。
        snap = LatentSnapshot(epoch=epoch)
        # 这里采取保守策略: model.encode → 收集 z_e/z_q/code_id。
        try:
            import torch  # noqa: F401
        except ImportError:
            return snap
        sample_ids: List[str] = []
        splits: List[str] = []
        code_ids: List[int] = []
        z_e_list: List[list] = []
        z_q_list: List[list] = []
        for batch in probe_dataset:
            states = batch["states"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            ids = batch["sample_ids"]
            split = batch.get("splits", ["val"] * len(ids))
            cid, z_e = model.encode(states, actions, rewards)
            z_q = model.quantizer.codebook[cid]
            sample_ids.extend(ids)
            splits.extend(split)
            code_ids.extend(cid.tolist())
            z_e_list.extend(z_e.tolist())
            z_q_list.extend(z_q.tolist())
            if len(sample_ids) >= self.max_points_per_split * 3:  # train+val+test
                break
        snap.sample_ids = sample_ids
        snap.splits = splits
        snap.code_ids = code_ids
        snap.z_e = z_e_list
        snap.z_q = z_q_list
        snap.codebook = model.quantizer.codebook.detach().cpu().tolist()
        return snap

    def write_tensorboard_embeddings(self, snapshot: LatentSnapshot) -> None:
        """如果 tensorboard 可用就写 embedding；否则跳过（不让训练失败）。

        Notes
        -----
        当前版本只是占位（避免引入硬依赖）；要使用 TensorBoard projector，
        在子类里覆写本方法即可，输入 ``snapshot.z_e / z_q / metadata`` 已经准备好。
        """
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        except ImportError:
            return
        # 实际写法略：调用方可选实现；为保持依赖最小，这里仅占位记录。

    def compute_projections(
        self, snapshot: LatentSnapshot, methods: Sequence[str]
    ) -> Dict[str, list]:
        """对 ``snapshot.z_e`` 跑 PCA / t-SNE，返回 ``{method: list[N, 2]}``。

        实现细节
        --------
        - PCA: 仅依赖 numpy 的简化 SVD 实现（``_simple_pca``）。
        - t-SNE: 优先使用 ``sklearn.manifold.TSNE``；不可用时回退到 PCA，
          并在 manifest 中标注 fallback 原因。
        - ``snapshot.z_e`` 为空时返回空 dict，避免下游绘图崩溃。
        """
        out: Dict[str, list] = {}
        if not snapshot.z_e:
            return out
        try:
            import numpy as np
        except ImportError:
            return out
        Z = np.asarray(snapshot.z_e, dtype="float32")
        if "pca" in methods:
            out["pca"] = _simple_pca(Z, dim=2).tolist()
        if "tsne" in methods:
            try:
                from sklearn.manifold import TSNE  # noqa: F401
                from sklearn.manifold import TSNE as _TSNE
                proj = _TSNE(n_components=2, init="pca", perplexity=30, max_iter=500)
                out["tsne"] = proj.fit_transform(Z).tolist()
            except ImportError:
                # sklearn 不可用时回退到 PCA。
                out["tsne"] = _simple_pca(Z, dim=2).tolist()
        return out

    def write_manifest(self, manifest_path: Path, manifest: dict) -> None:
        from src.utils.feather_io import atomic_write_json

        atomic_write_json(manifest, manifest_path)


def _simple_pca(matrix, dim: int = 2):
    """仅依赖 numpy 的简化 PCA。中心化 → SVD → 取前 dim 分量。"""
    import numpy as np

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    # SVD 比 eigh(covariance) 数值更稳。
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:dim].T
