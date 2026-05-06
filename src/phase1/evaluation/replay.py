"""Validation replay: teacher / student / boundary."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Sequence

from src.preprocess_data.horizon_builder import HorizonRecord
from src.trading.cost_model import LobDepthCostModel
from src.trading.env import HorizonInputs, TradingEnv


@dataclass
class HorizonReplayRecord:
    sample_id: str
    code_id: int
    teacher_actions: List[int] = field(default_factory=list)
    student_actions: List[int] = field(default_factory=list)
    teacher_step_returns: List[float] = field(default_factory=list)
    student_step_returns: List[float] = field(default_factory=list)
    teacher_net_return: float = 0.0
    student_net_return: float = 0.0
    cost_paid: float = 0.0
    teacher_reject_count: int = 0
    student_reject_count: int = 0


@dataclass
class BoundaryReplayResult:
    horizon_boundary_turnover_cost: float
    horizon_boundary_position_consistency: float


class Phase1ReplayEvaluator:
    """编排三种 replay；不持久化结果。"""

    def __init__(self, env_factory: Callable[[], TradingEnv]) -> None:
        self.env_factory = env_factory

    # ---------- teacher ----------

    def replay_dp_teacher(self, horizon: HorizonRecord) -> HorizonReplayRecord:
        """teacher actions 已存在 ``horizon.actions``（DP 阶段填好）。

        本方法只做一次 env replay，把 fee/slippage/reject 计入 ``HorizonReplayRecord``。
        ``code_id`` 用 ``-1`` 占位（teacher 不属于任何 archetype）。

        Raises
        ------
        RuntimeError : ``horizon.actions`` 为空（DP 阶段漏填）。
        """
        if horizon.actions is None:
            raise RuntimeError(
                f"horizon {horizon.sample_id} 缺少 DP actions; teacher replay 需要先跑 DP"
            )
        env = self.env_factory()
        env.reset(HorizonInputs(prices=list(horizon.prices), execution_books=list(horizon.execution_books)))
        rewards, infos = env.replay(horizon.actions)
        cost = sum(info.fee + info.slippage for info in infos)
        rejected = sum(1 for info in infos if info.rejected)
        return HorizonReplayRecord(
            sample_id=horizon.sample_id,
            code_id=-1,
            teacher_actions=list(horizon.actions),
            teacher_step_returns=list(rewards),
            teacher_net_return=sum(rewards),
            cost_paid=cost,
            teacher_reject_count=rejected,
        )

    # ---------- student ----------

    def replay_student_online(
        self,
        horizon: HorizonRecord,
        decoder,
        codebook,
        code_id: int,
    ) -> HorizonReplayRecord:
        """因果在线 replay。

        实现注意
        --------
        - 由于 decoder 已经是单向 LSTM（``ArchetypeDecoder.lstm.bidirectional=False``），
          整段 forward 等价于逐步前进；因果性由模型结构保证。
        - 为效率先一次性 forward 拿到 logits，再 ``argmax`` 得 actions，最后走 env replay。
        - 若未来把 decoder 换成 Transformer 等结构，必须在该方法内改为逐步推理，
          并加 causal mask 单测。

        Parameters
        ----------
        horizon : 已经跑过 DP 的 record（含 prices/execution_books/states）。
        decoder : 通常是 ``model.decoder``。
        codebook : ``model.quantizer.codebook``（``Tensor[K, code_dim]``）。
        code_id : 当前 horizon 选用的 archetype id；Phase I validation 直接来自 encoder。

        Returns
        -------
        HorizonReplayRecord : student 路径的 actions / step returns / cost / reject。
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError("student replay 需要 torch")

        env = self.env_factory()
        env.reset(HorizonInputs(prices=list(horizon.prices), execution_books=list(horizon.execution_books)))
        states = torch.tensor(horizon.states, dtype=torch.float32).unsqueeze(0)
        cb_tensor = codebook
        if not isinstance(cb_tensor, torch.Tensor):
            cb_tensor = torch.tensor(cb_tensor, dtype=torch.float32)
        _device = cb_tensor.device
        states = states.to(_device)
        z_q = cb_tensor[code_id].unsqueeze(0)
        decoder.eval()
        with torch.no_grad():
            logits = decoder(states, z_q)
        actions = logits.argmax(dim=-1).squeeze(0).tolist()
        rewards, infos = env.replay(actions)
        cost = sum(info.fee + info.slippage for info in infos)
        rejected = sum(1 for info in infos if info.rejected)
        return HorizonReplayRecord(
            sample_id=horizon.sample_id,
            code_id=int(code_id),
            student_actions=actions,
            student_step_returns=list(rewards),
            student_net_return=sum(rewards),
            cost_paid=cost,
            student_reject_count=rejected,
        )

    def replay_student_online_sequence(
        self,
        ordered_horizons: Sequence[HorizonRecord],
        decoder,
        codebook,
        code_ids: Sequence[int],
    ) -> List[HorizonReplayRecord]:
        """按时间顺序 replay student，并在 horizon 间继承仓位。

        该路径用于 Phase I causal online validation：code_id 由 teacher-free
        selector 给出，decoder 产生动作，env 逐 horizon reset 时继承上一段
        末仓位，边界换仓成本由第一步 ``TradingEnv.step`` 自然扣除。
        """
        if len(ordered_horizons) != len(code_ids):
            raise ValueError("ordered_horizons 与 code_ids 长度不一致")

        try:
            import torch
        except ImportError:
            raise RuntimeError("student online sequence replay 需要 torch")

        records: List[HorizonReplayRecord] = []
        prev_position = 0
        cb_tensor = codebook
        if not isinstance(cb_tensor, torch.Tensor):
            cb_tensor = torch.tensor(cb_tensor, dtype=torch.float32)
        _device = cb_tensor.device
        decoder.eval()
        for horizon, code_id in zip(ordered_horizons, code_ids):
            states = torch.tensor(horizon.states, dtype=torch.float32).unsqueeze(0).to(_device)
            z_q = cb_tensor[int(code_id)].unsqueeze(0)
            with torch.no_grad():
                logits = decoder(states, z_q)
            actions = logits.argmax(dim=-1).squeeze(0).tolist()

            env = self.env_factory()
            env.reset(
                HorizonInputs(
                    prices=list(horizon.prices),
                    execution_books=list(horizon.execution_books),
                ),
                initial_position=prev_position,
            )
            rewards, infos = env.replay(actions)
            cost = sum(info.fee + info.slippage for info in infos)
            rejected = sum(1 for info in infos if info.rejected)
            if infos:
                prev_position = infos[-1].filled_position
            records.append(
                HorizonReplayRecord(
                    sample_id=horizon.sample_id,
                    code_id=int(code_id),
                    student_actions=list(actions),
                    student_step_returns=list(rewards),
                    student_net_return=sum(rewards),
                    cost_paid=cost,
                    student_reject_count=rejected,
                )
            )
        return records

    # ---------- boundary ----------

    def evaluate_horizon_boundaries(
        self,
        ordered_horizons: Sequence[HorizonRecord],
        decoded_actions: Sequence[Sequence[int]],
    ) -> BoundaryReplayResult:
        """跨 horizon 模拟仓位继承。

        实现要点
        --------
        - 第 ``i+1`` 个 horizon ``reset(initial_position=last_position_of_i)``。
        - 第一步 target_position 与 inherited 不一致时，通过同一 ``CostModel``
          扣换仓成本；统计 ``horizon_boundary_turnover_cost`` 与
          ``horizon_boundary_position_consistency``。
        - 单 horizon 时直接返回 ``consistency=1.0, cost=0.0``。

        Returns
        -------
        BoundaryReplayResult
        """
        if len(ordered_horizons) != len(decoded_actions):
            raise ValueError("ordered_horizons 与 decoded_actions 长度不一致")
        if len(ordered_horizons) < 2:
            return BoundaryReplayResult(
                horizon_boundary_turnover_cost=0.0,
                horizon_boundary_position_consistency=1.0,
            )

        # 按论文，env 默认 max_position=1；从 env_factory 拿到一致配置。
        env = self.env_factory()
        cost_model: LobDepthCostModel = env.cost_model

        # 必须先跑 ordered_horizons[0] 拿到其末仓位作为第一个 boundary 的 prev_position；
        # 否则第一个 boundary 总会被误判为"prev=flat"。
        prev_position = 0
        first_actions = decoded_actions[0]
        if first_actions:
            env.reset(
                HorizonInputs(
                    prices=list(ordered_horizons[0].prices),
                    execution_books=list(ordered_horizons[0].execution_books),
                ),
                initial_position=0,
            )
            _, infos_first = env.replay(first_actions)
            if infos_first:
                prev_position = infos_first[-1].filled_position

        total_cost = 0.0
        consistent = 0
        boundaries = 0
        for i in range(1, len(ordered_horizons)):
            current = ordered_horizons[i]
            actions_i = decoded_actions[i]
            if not actions_i:
                continue
            target_first = (env.max_position) * (int(actions_i[0]) - 1)
            book_first = current.execution_books[0]
            if prev_position == target_first:
                consistent += 1
                boundaries += 1
                # 持仓一致 → 不需要换仓
                # 仍按一段 horizon 推进 prev_position
            else:
                result = cost_model.execute(
                    prev_position=prev_position,
                    target_position=target_first,
                    execution_book=book_first,
                )
                total_cost += result.cost
                boundaries += 1

            # 跑完整段 horizon 用于得到末仓位
            env.reset(
                HorizonInputs(prices=list(current.prices), execution_books=list(current.execution_books)),
                initial_position=(prev_position if prev_position == target_first else target_first),
            )
            _, infos = env.replay(actions_i)
            if infos:
                prev_position = infos[-1].filled_position
            else:
                prev_position = target_first

        return BoundaryReplayResult(
            horizon_boundary_turnover_cost=total_cost / max(boundaries, 1),
            horizon_boundary_position_consistency=consistent / max(boundaries, 1),
        )
