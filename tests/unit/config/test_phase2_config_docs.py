from dataclasses import fields

from src.config.phase2_config import (
    CostAlignmentCheckConfig,
    DeploymentLadderConfig,
    DistributionShiftConfig,
    EarlyStoppingConfig,
    EnvShardsConfig,
    ExecutionStressConfig,
    HorizonScheduleConfig,
    LiveRiskControlsConfig,
    NumericalSafetyConfig,
    OnlineActionThrottleConfig,
    PHASE2_CONFIG_FIELD_DOCS,
    PPOConfig,
    Phase1ArtifactsConfig,
    Phase2Config,
    Phase2SelectionPolicyConfig,
    ResumeConfig,
    RewardNormalizationConfig,
    RewardScalingConfig,
    RollingValidationConfig,
    SelectorNetworkConfig,
    StateDimBreakdownConfig,
)


_NESTED_CONFIG_TYPES = {
    "phase1_artifacts": Phase1ArtifactsConfig,
    "horizon_schedule": HorizonScheduleConfig,
    "selector_network": SelectorNetworkConfig,
    "ppo": PPOConfig,
    "selection_policy": Phase2SelectionPolicyConfig,
    "reward_scaling": RewardScalingConfig,
    "reward_normalization": RewardNormalizationConfig,
    "cost_alignment_check": CostAlignmentCheckConfig,
    "early_stopping": EarlyStoppingConfig,
    "resume": ResumeConfig,
    "deployment_ladder": DeploymentLadderConfig,
    "env_shards": EnvShardsConfig,
    "state_dim_breakdown": StateDimBreakdownConfig,
    "live_risk_controls": LiveRiskControlsConfig,
    "distribution_shift": DistributionShiftConfig,
    "execution_stress": ExecutionStressConfig,
    "rolling_validation": RollingValidationConfig,
    "online_action_throttle": OnlineActionThrottleConfig,
    "numerical_safety": NumericalSafetyConfig,
}


def _phase2_field_paths(config_type, prefix=""):
    for config_field in fields(config_type):
        path = f"{prefix}.{config_field.name}" if prefix else config_field.name
        yield path
        nested_type = _NESTED_CONFIG_TYPES.get(path)
        if nested_type is not None:
            yield from _phase2_field_paths(nested_type, path)


def test_phase2_config_field_docs_cover_every_yaml_field():
    expected_paths = set(_phase2_field_paths(Phase2Config))

    assert set(PHASE2_CONFIG_FIELD_DOCS) == expected_paths


def test_phase2_config_field_docs_explain_why_and_tuning_effect():
    for path, doc in PHASE2_CONFIG_FIELD_DOCS.items():
        assert doc["why"].strip(), path
        assert doc["tuning_effect"].strip(), path
