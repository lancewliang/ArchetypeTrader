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
    RolloutCollectionConfig,
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
    "rollout_collection": RolloutCollectionConfig,
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


def test_phase2_config_from_dict_rollout_collection_defaults_and_explicit_values():
    default_config = Phase2Config.from_dict({})
    assert default_config.rollout_collection.mode == "serial"

    threaded_config = Phase2Config.from_dict({
        "rollout_collection": {
            "mode": "process",
            "max_workers": 2,
            "fail_fast": False,
            "process_start_method": "spawn",
            "worker_device": "cpu",
            "worker_startup_timeout_seconds": 15.0,
            "worker_step_timeout_seconds": 5.0,
            "restart_failed_workers": False,
            "shared_dataset_mode": "pickle",
        }
    })
    assert threaded_config.rollout_collection.mode == "process"
    assert threaded_config.rollout_collection.max_workers == 2
    assert threaded_config.rollout_collection.fail_fast is False
    assert threaded_config.rollout_collection.worker_step_timeout_seconds == 5.0
