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
    ckpt_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    expected_state_dim = config.get_state_dim(pair)

    ckpt_state_dim = int(ckpt_config.get("state_dim", expected_state_dim))
    ckpt_action_dim = int(ckpt_config.get("action_dim", config.action_dim))
    ckpt_latent_dim = int(ckpt_config.get("latent_dim", config.latent_dim))
    ckpt_num_archetypes = int(ckpt_config.get("num_archetypes", config.num_archetypes))
    ckpt_lstm_hidden_dim = int(ckpt_config.get("lstm_hidden_dim", config.lstm_hidden_dim))
    ckpt_decoder_arch = str(ckpt_config.get("decoder_arch", "bilstm"))
    ckpt_transformer_layers = int(
        ckpt_config.get(
            "decoder_transformer_layers",
            getattr(config, "decoder_transformer_layers", 2),
        ),
    )
    ckpt_transformer_heads = int(
        ckpt_config.get(
            "decoder_transformer_heads",
            getattr(config, "decoder_transformer_heads", 4),
        ),
    )
    ckpt_transformer_ffn_dim_raw = ckpt_config.get(
        "decoder_transformer_ffn_dim",
        getattr(config, "decoder_transformer_ffn_dim", None),
    )
    ckpt_transformer_dropout = float(
        ckpt_config.get(
            "decoder_transformer_dropout",
            getattr(config, "decoder_transformer_dropout", 0.0),
        ),
    )

    if ckpt_state_dim != expected_state_dim:
        raise ValueError(
            "Phase I checkpoint 的 state_dim 与当前配置不一致，无法加载并用于推理。\n"
            f"  checkpoint_state_dim={ckpt_state_dim}, config_state_dim={expected_state_dim}\n"
            "  提示: 请用与训练一致的 --cycle-feature-sets 运行 evaluate/train_phase2/train_phase3。",
        )
    if ckpt_action_dim != config.action_dim:
        raise ValueError(
            "Phase I checkpoint 的 action_dim 与当前配置不一致，无法加载并用于推理。\n"
            f"  checkpoint_action_dim={ckpt_action_dim}, config_action_dim={config.action_dim}",
        )

    codebook = VQCodebook(
        num_codes=ckpt_num_archetypes, code_dim=ckpt_latent_dim,
    ).to(device)
    codebook.load_state_dict(checkpoint["codebook"])

    decoder = VQDecoder(
        state_dim=ckpt_state_dim,
        code_dim=ckpt_latent_dim,
        hidden_dim=ckpt_lstm_hidden_dim,
        action_dim=ckpt_action_dim,
        decoder_arch=ckpt_decoder_arch,
        transformer_layers=ckpt_transformer_layers,
        transformer_heads=ckpt_transformer_heads,
        transformer_ffn_dim=(
            int(ckpt_transformer_ffn_dim_raw)
            if ckpt_transformer_ffn_dim_raw is not None
            else None
        ),
        transformer_dropout=ckpt_transformer_dropout,
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

    agent_state = checkpoint.get("agent", None)
    if not isinstance(agent_state, dict):
        raise ValueError(f"Phase II checkpoint 缺少 agent state_dict: {model_path}")

    try:
        shared0_w = agent_state["shared.0.weight"]
        shared2_w = agent_state["shared.2.weight"]
        policy_w = agent_state["policy_head.weight"]
    except KeyError as e:
        raise ValueError(f"Phase II agent state_dict 缺少关键参数 {e!s}: {model_path}") from e

    inferred_state_dim = int(shared0_w.shape[1])
    inferred_hidden_dim = int(shared0_w.shape[0])
    inferred_bottleneck_dim = int(shared2_w.shape[0])
    inferred_num_archetypes = int(policy_w.shape[0])
    expected_state_dim = config.get_state_dim(pair)

    if inferred_state_dim != expected_state_dim:
        raise ValueError(
            "Phase II checkpoint 的 state_dim 与当前配置不一致，无法加载并用于推理。\n"
            f"  checkpoint_state_dim={inferred_state_dim}, config_state_dim={expected_state_dim}\n"
            "  提示: 请用与训练一致的 --cycle-feature-sets 运行 evaluate。",
        )

    agent = SelectionAgent(
        state_dim=inferred_state_dim,
        num_archetypes=inferred_num_archetypes,
        hidden_dim=inferred_hidden_dim,
        bottleneck_dim=inferred_bottleneck_dim,
    ).to(device)
    agent.load_state_dict(agent_state)

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

    ckpt_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    expected_state_dim = config.get_state_dim(pair)
    ckpt_state_dim = int(ckpt_config.get("state_dim", expected_state_dim))
    ckpt_latent_dim = int(ckpt_config.get("latent_dim", config.latent_dim))
    ckpt_hidden_dim = int(ckpt_config.get("refinement_hidden_dim", config.refinement_hidden_dim))

    if ckpt_state_dim != expected_state_dim:
        raise ValueError(
            "Phase III checkpoint 的 state_dim 与当前配置不一致，无法加载并用于推理。\n"
            f"  checkpoint_state_dim={ckpt_state_dim}, config_state_dim={expected_state_dim}\n"
            "  提示: 请用与训练一致的 --cycle-feature-sets 运行 evaluate。",
        )

    context_dim = ckpt_latent_dim + 3
    agent = RefinementAgent(
        market_dim=ckpt_state_dim, context_dim=context_dim,
        hidden_dim=ckpt_hidden_dim,
    ).to(device)
    agent.load_state_dict(checkpoint["agent"])

    for p in agent.parameters():
        p.requires_grad = False
    agent.eval()
    return agent
