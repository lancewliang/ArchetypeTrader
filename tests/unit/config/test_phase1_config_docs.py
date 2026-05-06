from dataclasses import fields

from src.phase1.config import (
    BehaviorGuardrailConfig,
    CausalOnlineValidationConfig,
    CodebookConfig,
    CodebookHealthConfig,
    CodebookLocalOptimumEscapeConfig,
    CostConfig,
    DPConfig,
    DataAugmentationConfig,
    DiagnosticsConfig,
    EncoderInputConfig,
    ModelConfig,
    NoTradeCodeHealthConfig,
    NoTradeControlConfig,
    PHASE1_CONFIG_FIELD_DOCS,
    Phase1Config,
    RejectTransitionHealthConfig,
    RiskGuardrailConfig,
    SamplingHealthConfig,
    SelectionPolicyConfig,
    StratificationConfig,
    SyntheticHorizonConfig,
    TeacherQualityGuardrailConfig,
    TemporalContrastiveConfig,
    TrainingConfig,
)


_NESTED_CONFIG_TYPES = {
    "stratification": StratificationConfig,
    "data_augmentation": DataAugmentationConfig,
    "data_augmentation.temporal_contrastive": TemporalContrastiveConfig,
    "data_augmentation.synthetic_horizon": SyntheticHorizonConfig,
    "dp": DPConfig,
    "dp.cost_config": CostConfig,
    "dp.cost_config.reject_transition_health": RejectTransitionHealthConfig,
    "model": ModelConfig,
    "model.encoder_input": EncoderInputConfig,
    "model.codebook": CodebookConfig,
    "model.codebook.health": CodebookHealthConfig,
    "model.codebook.health.local_optimum_escape": CodebookLocalOptimumEscapeConfig,
    "training": TrainingConfig,
    "selection_policy": SelectionPolicyConfig,
    "selection_policy.risk": RiskGuardrailConfig,
    "selection_policy.behavior": BehaviorGuardrailConfig,
    "selection_policy.teacher": TeacherQualityGuardrailConfig,
    "selection_policy.online_validation": CausalOnlineValidationConfig,
    "diagnostics": DiagnosticsConfig,
}


def _phase1_field_paths(config_type, prefix=""):
    for config_field in fields(config_type):
        path = f"{prefix}.{config_field.name}" if prefix else config_field.name
        yield path
        nested_type = _NESTED_CONFIG_TYPES.get(path)
        if nested_type is not None:
            yield from _phase1_field_paths(nested_type, path)


def test_phase1_config_field_docs_cover_every_yaml_field():
    expected_paths = set(_phase1_field_paths(Phase1Config))

    assert set(PHASE1_CONFIG_FIELD_DOCS) == expected_paths


def test_phase1_config_field_docs_explain_why_and_tuning_effect():
    for path, doc in PHASE1_CONFIG_FIELD_DOCS.items():
        assert doc["why"].strip(), path
        assert doc["tuning_effect"].strip(), path
