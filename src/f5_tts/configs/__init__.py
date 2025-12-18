"""
Model configuration loading utilities.

This module provides utilities for loading and managing model configurations
from YAML files.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Dict, Any, List

from omegaconf import OmegaConf


def get_config_path(model_name: str) -> Path:
    """
    Get path to model configuration YAML file.

    Args:
        model_name: Name of the model (e.g., 'F5TTS_v1_Base')

    Returns:
        Path to the configuration file

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
    """
    config_path = Path(str(files("f5_tts").joinpath(f"configs/{model_name}.yaml")))
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    return config_path


def load_model_config(model_name: str) -> Dict[str, Any]:
    """
    Load model configuration from YAML file.

    Args:
        model_name: Name of the model (e.g., 'F5TTS_v1_Base')

    Returns:
        Dictionary containing model configuration
    """
    config_path = get_config_path(model_name)
    return OmegaConf.to_container(OmegaConf.load(str(config_path)), resolve=True)


def get_available_models() -> List[str]:
    """
    List all available model configurations.

    Returns:
        List of model names with available configurations
    """
    config_dir = Path(str(files("f5_tts").joinpath("configs")))
    return [f.stem for f in config_dir.glob("*.yaml")]


__all__ = [
    "get_config_path",
    "load_model_config",
    "get_available_models",
]
