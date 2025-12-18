"""
Central location for all project constants and default values.

This module consolidates all hardcoded values from across the codebase
to provide a single source of truth for configuration.
"""

from __future__ import annotations

# =============================================================================
# Audio Processing Constants
# =============================================================================

TARGET_SAMPLE_RATE = 24000
N_MEL_CHANNELS = 100
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_FFT = 1024

# =============================================================================
# Inference Defaults
# =============================================================================

DEFAULT_ODE_METHOD = "euler"
DEFAULT_NFE_STEP = 32
DEFAULT_CFG_STRENGTH = 2.0
DEFAULT_SWAY_SAMPLING_COEF = -1.0
DEFAULT_SPEED = 1.0
DEFAULT_TARGET_RMS = 0.1
DEFAULT_CROSS_FADE_DURATION = 0.15
DEFAULT_FIX_DURATION = None

# Text chunking
DEFAULT_MAX_CHARS = 135

# =============================================================================
# Training Defaults
# =============================================================================

DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_BATCH_SIZE_PER_GPU = 3200
DEFAULT_BATCH_SIZE_TYPE = "frame"
DEFAULT_MAX_SAMPLES = 64
DEFAULT_GRAD_ACCUMULATION_STEPS = 1
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_EPOCHS = 1000
DEFAULT_NUM_WARMUP_UPDATES = 300
DEFAULT_SAVE_PER_UPDATES = 10000
DEFAULT_LAST_PER_UPDATES = 50000
DEFAULT_KEEP_LAST_N_CHECKPOINTS = -1

# =============================================================================
# Model Configurations
# =============================================================================

MODEL_CONFIGS = {
    "F5TTS_v1_Base": {
        "backbone": "DiT",
        "dim": 1024,
        "depth": 22,
        "heads": 16,
        "ff_mult": 2,
        "text_dim": 512,
        "conv_layers": 4,
        "text_mask_padding": True,
    },
    "F5TTS_Base": {
        "backbone": "DiT",
        "dim": 1024,
        "depth": 22,
        "heads": 16,
        "ff_mult": 2,
        "text_dim": 512,
        "text_mask_padding": False,
        "conv_layers": 4,
        "pe_attn_head": 1,
    },
    "F5TTS_Small": {
        "backbone": "DiT",
        "dim": 768,
        "depth": 18,
        "heads": 12,
        "ff_mult": 2,
        "text_dim": 512,
        "conv_layers": 4,
    },
    "E2TTS_Base": {
        "backbone": "UNetT",
        "dim": 1024,
        "depth": 24,
        "heads": 16,
        "ff_mult": 4,
        "text_mask_padding": False,
        "pe_attn_head": 1,
    },
    "E2TTS_Small": {
        "backbone": "UNetT",
        "dim": 768,
        "depth": 20,
        "heads": 12,
        "ff_mult": 4,
    },
}

# =============================================================================
# Pretrained Model Paths (HuggingFace)
# =============================================================================

PRETRAINED_MODELS = {
    "F5TTS_v1_Base": "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "F5TTS_Base": "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt",
    "E2TTS_Base": "hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.pt",
}

# =============================================================================
# Vocoder Configuration
# =============================================================================

MEL_SPEC_TYPES = ["vocos", "bigvgan"]
DEFAULT_MEL_SPEC_TYPE = "vocos"

VOCOS_REPO_ID = "charactr/vocos-mel-24khz"
BIGVGAN_REPO_ID = "nvidia/bigvgan_v2_24khz_100band_256x"

# =============================================================================
# Tokenizer Configuration
# =============================================================================

TOKENIZER_TYPES = ["pinyin", "char", "byte", "custom"]
DEFAULT_TOKENIZER = "pinyin"

# =============================================================================
# Model Architecture Constants
# =============================================================================

# Maximum precomputed position for rotary embeddings (~44s of 24khz audio)
PRECOMPUTE_MAX_POS = 4096

# Default dimension for timestep embedding
DEFAULT_FREQ_EMBED_DIM = 256

# =============================================================================
# Available Experiment Names
# =============================================================================

EXPERIMENT_NAMES = ["F5TTS_v1_Base", "F5TTS_Base", "F5TTS_Small", "E2TTS_Base", "E2TTS_Small"]

# =============================================================================
# Logger Options
# =============================================================================

LOGGER_TYPES = ["wandb", "tensorboard"]
