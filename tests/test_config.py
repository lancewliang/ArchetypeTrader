"""Tests for src/config.py — Config dataclass and CLI argument parsing."""

from src.config import Config, parse_args


class TestConfigDefaults:
    """Verify all default values match the design document."""

    def test_data_paths(self):
        cfg = Config()
        assert cfg.data_dir == "data/feature_list"
        assert cfg.result_dir == "result"
        assert cfg.pairs == ["BTC", "ETH", "DOT", "BNB"]

    def test_feature_dims(self):
        cfg = Config()
        assert cfg.single_feature_dim == 36
        assert cfg.trend_feature_dim == 9
        assert cfg.state_dim == 45

    def test_mdp_config(self):
        cfg = Config()
        assert cfg.action_dim == 3
        assert cfg.horizon == 72
        assert cfg.commission_rate == 0.0002
        assert cfg.max_positions == {
            "BTC": 8, "ETH": 100, "DOT": 2500, "BNB": 200
        }

    def test_phase1_config(self):
        cfg = Config()
        assert cfg.lstm_hidden_dim == 128
        assert cfg.latent_dim == 16
        assert cfg.num_archetypes == 10
        assert cfg.vq_beta0 == 0.25
        assert cfg.num_trajectories == 30000
        assert cfg.phase1_epochs == 100

    def test_phase2_config(self):
        cfg = Config()
        assert cfg.phase2_total_steps == 3_000_000
        assert cfg.selection_alpha == 1.0

    def test_phase3_config(self):
        cfg = Config()
        assert cfg.phase3_total_steps == 1_000_000
        assert cfg.refinement_beta1 == 0.5
        assert cfg.refinement_beta2 == 1.0

    def test_training_config(self):
        cfg = Config()
        assert cfg.discount_factor == 0.99
        assert cfg.learning_rate == 3e-4
        assert cfg.batch_size == 256

    def test_data_splits(self):
        cfg = Config()
        assert cfg.train_start == "2021-06-01"
        assert cfg.train_end == "2023-05-31"
        assert cfg.val_start == "2023-06-01"
        assert cfg.val_end == "2023-12-31"
        assert cfg.test_start == "2024-01-01"
        assert cfg.test_end == "2024-09-01"

    def test_annualization_factor(self):
        cfg = Config()
        assert cfg.annualization_factor == 52560


class TestParseArgs:
    """Verify CLI argument overrides work correctly."""

    def test_no_args_returns_defaults(self):
        cfg = parse_args([])
        assert cfg.learning_rate == 3e-4
        assert cfg.batch_size == 256
        assert cfg.pairs == ["BTC", "ETH", "DOT", "BNB"]

    def test_pair_override(self):
        cfg = parse_args(["--pair", "BTC"])
        assert cfg.pairs == ["BTC"]

    def test_learning_rate_override(self):
        cfg = parse_args(["--lr", "1e-3"])
        assert cfg.learning_rate == 1e-3

    def test_batch_size_override(self):
        cfg = parse_args(["--batch-size", "512"])
        assert cfg.batch_size == 512

    def test_beta1_override(self):
        cfg = parse_args(["--beta1", "0.3"])
        assert cfg.refinement_beta1 == 0.3

    def test_beta2_override(self):
        cfg = parse_args(["--beta2", "2.0"])
        assert cfg.refinement_beta2 == 2.0

    def test_horizon_override(self):
        cfg = parse_args(["--horizon", "36"])
        assert cfg.horizon == 36

    def test_data_dir_override(self):
        cfg = parse_args(["--data-dir", "/tmp/data"])
        assert cfg.data_dir == "/tmp/data"

    def test_multiple_overrides(self):
        cfg = parse_args([
            "--pair", "ETH",
            "--lr", "1e-2",
            "--batch-size", "128",
            "--beta1", "0.7",
        ])
        assert cfg.pairs == ["ETH"]
        assert cfg.learning_rate == 1e-2
        assert cfg.batch_size == 128
        assert cfg.refinement_beta1 == 0.7
        # Non-overridden values stay default
        assert cfg.discount_factor == 0.99
        assert cfg.num_archetypes == 10

    def test_phase_steps_override(self):
        cfg = parse_args([
            "--phase2-total-steps", "100000",
            "--phase3-total-steps", "50000",
        ])
        assert cfg.phase2_total_steps == 100000
        assert cfg.phase3_total_steps == 50000

    def test_commission_rate_override(self):
        cfg = parse_args(["--commission-rate", "0.001"])
        assert cfg.commission_rate == 0.001

    def test_num_trajectories_override(self):
        cfg = parse_args(["--num-trajectories", "1000"])
        assert cfg.num_trajectories == 1000

    def test_phase1_epochs_override(self):
        cfg = parse_args(["--phase1-epochs", "50"])
        assert cfg.phase1_epochs == 50


class TestConfigFromArgs:
    """Test Config.from_args classmethod directly."""

    def test_from_args_ignores_none(self):
        """None values in namespace should not override defaults."""
        import argparse
        ns = argparse.Namespace(learning_rate=None, batch_size=None)
        cfg = Config.from_args(ns)
        assert cfg.learning_rate == 3e-4
        assert cfg.batch_size == 256

    def test_from_args_applies_values(self):
        import argparse
        ns = argparse.Namespace(learning_rate=0.01, batch_size=64)
        cfg = Config.from_args(ns)
        assert cfg.learning_rate == 0.01
        assert cfg.batch_size == 64
