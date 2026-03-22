"""Tests for src/utils/logger.py — unified logging utility."""

import logging

from src.utils.logger import get_logger


class TestGetLogger:
    """Verify get_logger returns a properly configured logger."""

    def test_returns_logger_instance(self):
        logger = get_logger("test.basic")
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self):
        logger = get_logger("test.name")
        assert logger.name == "test.name"

    def test_default_level_is_info(self):
        logger = get_logger("test.level")
        assert logger.level == logging.INFO

    def test_custom_level(self):
        logger = get_logger("test.custom_level", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_has_stream_handler(self):
        logger = get_logger("test.handler")
        assert len(logger.handlers) >= 1
        assert any(
            isinstance(h, logging.StreamHandler) for h in logger.handlers
        )

    def test_no_duplicate_handlers(self):
        """Calling get_logger twice with the same name should not add extra handlers."""
        name = "test.no_dup"
        logger1 = get_logger(name)
        count = len(logger1.handlers)
        logger2 = get_logger(name)
        assert logger2 is logger1
        assert len(logger2.handlers) == count

    def test_log_output(self, capfd):
        """Logger should write formatted messages to stderr."""
        logger = get_logger("test.output")
        logger.warning("hello %s", "world")
        captured = capfd.readouterr()
        assert "hello world" in captured.err
        assert "[WARNING]" in captured.err
        assert "test.output" in captured.err

    def test_info_level_filters_debug(self, capfd):
        """DEBUG messages should be suppressed at INFO level."""
        logger = get_logger("test.filter")
        logger.debug("should not appear")
        captured = capfd.readouterr()
        assert "should not appear" not in captured.err

    def test_training_metrics_logging(self, capfd):
        """Simulate training metric output (loss, reward)."""
        logger = get_logger("test.metrics")
        logger.info("epoch=5 loss=0.0312 reward=1.45")
        captured = capfd.readouterr()
        assert "loss=0.0312" in captured.err
        assert "reward=1.45" in captured.err
