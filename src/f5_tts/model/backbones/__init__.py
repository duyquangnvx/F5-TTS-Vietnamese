"""
Backbone transformer architectures for F5-TTS.

This module provides different transformer backbone architectures:
- DiT: Diffusion Transformer (default for F5-TTS)
- UNetT: Flat U-Net Transformer (used by E2-TTS)
- MMDiT: Multimodal Diffusion Transformer (experimental)
"""

from f5_tts.model.backbones.dit import DiT
from f5_tts.model.backbones.unett import UNetT
from f5_tts.model.backbones.mmdit import MMDiT

__all__ = [
    "DiT",
    "UNetT",
    "MMDiT",
]
