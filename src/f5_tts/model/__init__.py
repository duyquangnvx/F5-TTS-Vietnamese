"""
F5-TTS Model Components.

This module provides the core model components for F5-TTS:
- CFM: Conditional Flow Matching model
- DiT, UNetT, MMDiT: Transformer backbone architectures
- Trainer: Training orchestration
"""

from f5_tts.model.cfm import CFM
from f5_tts.model.backbones import DiT, UNetT, MMDiT
from f5_tts.model.trainer import Trainer
from f5_tts.model.dataset import load_dataset, HFDataset, CustomDataset


__all__ = [
    # Core model
    "CFM",
    # Backbones
    "DiT",
    "UNetT",
    "MMDiT",
    # Training
    "Trainer",
    # Dataset
    "load_dataset",
    "HFDataset",
    "CustomDataset",
]
