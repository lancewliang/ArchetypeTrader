"""Phase I 模型加载模块

功能说明:
    负责加载 Phase I 训练好的编码器、码本、解码器和归一化器，
    并将它们冻结用于 Phase II 训练。
"""

from __future__ import annotations

import os
from typing import Any

import torch

from src.phase1.codebook import VQCodebook
from src.phase1.vq_decoder import VQDecoder
from src.phase1.vq_encoder import VQEncoder
from src.utils.logger import get_logger
from src.utils.normalizer import StateNormalizer

logger = get_logger(__name__)


def load_phase1_model(config: Any, pair: str, device: torch.device):
    """加载 Phase I 模型（编码器 + 码本 + 冻结 Decoder）+ 归一化统计量。

    Returns:
        encoder, codebook, decoder, normalizer
    """
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

    ckpt_state_dim = int(ckpt_config.get("state_dim", config.state_dim))
    ckpt_action_dim = int(ckpt_config.get("action_dim", config.action_dim))
    ckpt_latent_dim = int(ckpt_config.get("latent_dim", config.latent_dim))
    ckpt_num_archetypes = int(ckpt_config.get("num_archetypes", config.num_archetypes))
    ckpt_lstm_hidden_dim = int(ckpt_config.get("lstm_hidden_dim", config.lstm_hidden_dim))

    if ckpt_state_dim != config.state_dim:
        logger.warning(
            "Phase I checkpoint state_dim=%d 与当前 config.state_dim=%d 不一致；"
            "请确认 --cycle-feature-sets 与训练时一致。",
            ckpt_state_dim, config.state_dim,
        )
    if ckpt_action_dim != config.action_dim:
        logger.warning(
            "Phase I checkpoint action_dim=%d 与当前 config.action_dim=%d 不一致。",
            ckpt_action_dim, config.action_dim,
        )

    encoder = VQEncoder(
        state_dim=ckpt_state_dim,
        action_dim=ckpt_action_dim,
        hidden_dim=ckpt_lstm_hidden_dim,
        latent_dim=ckpt_latent_dim,
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"])

    codebook = VQCodebook(
        num_codes=ckpt_num_archetypes,
        code_dim=ckpt_latent_dim,
    ).to(device)
    codebook.load_state_dict(checkpoint["codebook"])

    decoder = VQDecoder(
        state_dim=ckpt_state_dim,
        code_dim=ckpt_latent_dim,
        hidden_dim=ckpt_lstm_hidden_dim,
        action_dim=ckpt_action_dim,
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder"])

    for param in encoder.parameters():
        param.requires_grad = False
    for param in codebook.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False

    encoder.eval()
    codebook.eval()
    decoder.eval()

    normalizer = StateNormalizer.from_checkpoint_dict(checkpoint)
    if normalizer is not None:
        logger.info("Phase I 归一化统计量已加载")
    else:
        logger.warning("Phase I checkpoint 中无 norm_stats，跳过归一化")

    logger.info("Phase I 模型加载完成，Encoder、Codebook 和 Decoder 已冻结")
    return encoder, codebook, decoder, normalizer
