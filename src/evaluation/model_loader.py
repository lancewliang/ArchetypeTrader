"""ModelLoader — 三阶段模型加载

从 evaluate.py 中抽取 load_phase1/2/3_model，集中管理模型加载逻辑。
"""

from __future__ import annotations

import os
from typing import Tuple

import torch

from src.config import Config, parse_args
from src.phase1.codebook import VQCodebook
from src.phase1.vq_decoder import VQDecoder
from src.phase2.selection_agent import SelectionAgent
from src.phase3.refinement_agent import RefinementAgent
from src.utils.logger import get_logger
from src.utils.normalizer import StateNormalizer

logger = get_logger(__name__)


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_phase1_model(
    config: Config | None = None,
    pair: str = "ETH",
    device: torch.device | None = None,
) -> Tuple[VQCodebook, VQDecoder, StateNormalizer | None]:
    """加载 Phase I 模型（码本 + 冻结 Decoder）+ 归一化统计量。"""
    if config is None:
        config = parse_args([])
    if device is None:
        device = _default_device()

    model_path = os.path.join(
        config.get_stage_result_dir(pair, "phase1_archetype_discovery"),
        f"{pair}_vq_model.pt",
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Phase I 模型文件不存在: {model_path}\n"
            f"请先运行 Phase I 训练: python scripts/train_phase1.py --pair {pair}"
        )

    logger.info("加载 Phase I 模型: %s", model_path)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    codebook = VQCodebook(
        num_codes=config.num_archetypes, code_dim=config.latent_dim,
    ).to(device)
    codebook.load_state_dict(checkpoint["codebook"])

    decoder = VQDecoder(
        state_dim=config.state_dim,
        code_dim=config.latent_dim,
        hidden_dim=config.lstm_hidden_dim,
        action_dim=config.action_dim,
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder"])

    for p in codebook.parameters():
        p.requires_grad = False
    for p in decoder.parameters():
        p.requires_grad = False
    codebook.eval()
    decoder.eval()

    normalizer = StateNormalizer.from_checkpoint_dict(checkpoint)
    if normalizer is not None:
        logger.info("Phase I 归一化统计量已加载")

    return codebook, decoder, normalizer


def load_phase2_model(
    config: Config | None = None,
    pair: str = "ETH",
    device: torch.device | None = None,
) -> SelectionAgent:
    """加载 Phase II 模型（冻结 SelectionAgent）。"""
    if config is None:
        config = parse_args([])
    if device is None:
        device = _default_device()

    model_path = os.path.join(
        config.get_stage_result_dir(pair, "phase2_archetype_selection"),
        f"{pair}_selection_agent.pt",
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Phase II 模型文件不存在: {model_path}\n"
            f"请先运行 Phase II 训练: python scripts/train_phase2.py --pair {pair}"
        )

    logger.info("加载 Phase II 模型: %s", model_path)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    agent = SelectionAgent(
        state_dim=config.state_dim, num_archetypes=config.num_archetypes,
        hidden_dim=config.phase2_hidden_dim,
        bottleneck_dim=config.phase2_bottleneck_dim,
    ).to(device)
    agent.load_state_dict(checkpoint["agent"])

    for p in agent.parameters():
        p.requires_grad = False
    agent.eval()
    return agent


def load_phase3_model(
    config: Config | None = None,
    pair: str = "ETH",
    device: torch.device | None = None,
) -> RefinementAgent:
    """加载 Phase III 模型（RefinementAgent）。"""
    if config is None:
        config = parse_args([])
    if device is None:
        device = _default_device()

    beta1 = config.refinement_beta1
    model_path = os.path.join(
        config.get_stage_result_dir(pair, "phase3_archetype_refinement"),
        f"{pair}_refinement_agent_beta{beta1}.pt",
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Phase III 模型文件不存在: {model_path}\n"
            f"请先运行 Phase III 训练: python scripts/train_phase3.py --pair {pair}"
        )

    logger.info("加载 Phase III 模型: %s", model_path)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    context_dim = config.latent_dim + 3
    agent = RefinementAgent(
        market_dim=config.state_dim, context_dim=context_dim,
        hidden_dim=config.refinement_hidden_dim,
    ).to(device)
    agent.load_state_dict(checkpoint["agent"])

    for p in agent.parameters():
        p.requires_grad = False
    agent.eval()
    return agent
