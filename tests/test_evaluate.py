import pytest
import os
from src.config import config


def test_reports_dir_exists():
    """The reports directory should be created on import."""
    assert os.path.isdir(config.paths.reports_dir)


def test_model_dir_exists():
    """The models directory should be created on import."""
    assert os.path.isdir(config.paths.model_dir)


def test_data_dir_exists():
    """The data directory should be created on import."""
    assert os.path.isdir(config.paths.data_dir)
