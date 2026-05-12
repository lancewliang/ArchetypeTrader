import os

from src.utils.runtime import RuntimeUtils


def test_init_random_seed_sets_cublas_workspace_config(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    seed_status = RuntimeUtils.init_random_seed(42, deterministic=True)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert seed_status["cublas_workspace_config"] == ":4096:8"


def test_init_random_seed_preserves_existing_cublas_workspace_config(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    seed_status = RuntimeUtils.init_random_seed(42, deterministic=True)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert seed_status["cublas_workspace_config"] == ":16:8"
