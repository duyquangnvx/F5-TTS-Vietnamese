"""
Tests for the constants module.
"""

from __future__ import annotations

import pytest

from f5_tts import constants


class TestAudioConstants:
    """Tests for audio processing constants."""

    def test_target_sample_rate(self):
        """Test that target sample rate is defined and valid."""
        assert constants.TARGET_SAMPLE_RATE == 24000
        assert isinstance(constants.TARGET_SAMPLE_RATE, int)

    def test_n_mel_channels(self):
        """Test that mel channels is defined and valid."""
        assert constants.N_MEL_CHANNELS == 100
        assert isinstance(constants.N_MEL_CHANNELS, int)

    def test_hop_length(self):
        """Test that hop length is defined and valid."""
        assert constants.HOP_LENGTH == 256
        assert isinstance(constants.HOP_LENGTH, int)

    def test_win_length(self):
        """Test that window length is defined and valid."""
        assert constants.WIN_LENGTH == 1024
        assert isinstance(constants.WIN_LENGTH, int)

    def test_n_fft(self):
        """Test that FFT size is defined and valid."""
        assert constants.N_FFT == 1024
        assert isinstance(constants.N_FFT, int)


class TestInferenceDefaults:
    """Tests for inference default values."""

    def test_nfe_step(self):
        """Test NFE step default."""
        assert constants.DEFAULT_NFE_STEP == 32
        assert constants.DEFAULT_NFE_STEP > 0

    def test_cfg_strength(self):
        """Test CFG strength default."""
        assert constants.DEFAULT_CFG_STRENGTH == 2.0
        assert constants.DEFAULT_CFG_STRENGTH > 0

    def test_speed(self):
        """Test speed default."""
        assert constants.DEFAULT_SPEED == 1.0
        assert constants.DEFAULT_SPEED > 0

    def test_target_rms(self):
        """Test target RMS default."""
        assert constants.DEFAULT_TARGET_RMS == 0.1
        assert 0 < constants.DEFAULT_TARGET_RMS < 1

    def test_cross_fade_duration(self):
        """Test cross-fade duration default."""
        assert constants.DEFAULT_CROSS_FADE_DURATION == 0.15
        assert constants.DEFAULT_CROSS_FADE_DURATION >= 0


class TestModelConfigs:
    """Tests for model configuration dictionaries."""

    def test_model_configs_exist(self):
        """Test that model configs dictionary exists."""
        assert hasattr(constants, "MODEL_CONFIGS")
        assert isinstance(constants.MODEL_CONFIGS, dict)

    def test_f5tts_v1_base_config(self):
        """Test F5TTS_v1_Base configuration."""
        assert "F5TTS_v1_Base" in constants.MODEL_CONFIGS
        config = constants.MODEL_CONFIGS["F5TTS_v1_Base"]
        assert config["backbone"] == "DiT"
        assert config["dim"] == 1024
        assert config["depth"] == 22
        assert config["heads"] == 16

    def test_f5tts_base_config(self):
        """Test F5TTS_Base configuration."""
        assert "F5TTS_Base" in constants.MODEL_CONFIGS
        config = constants.MODEL_CONFIGS["F5TTS_Base"]
        assert config["backbone"] == "DiT"

    def test_e2tts_base_config(self):
        """Test E2TTS_Base configuration."""
        assert "E2TTS_Base" in constants.MODEL_CONFIGS
        config = constants.MODEL_CONFIGS["E2TTS_Base"]
        assert config["backbone"] == "UNetT"


class TestPretrainedModels:
    """Tests for pretrained model paths."""

    def test_pretrained_models_exist(self):
        """Test that pretrained models dictionary exists."""
        assert hasattr(constants, "PRETRAINED_MODELS")
        assert isinstance(constants.PRETRAINED_MODELS, dict)

    def test_pretrained_model_paths_format(self):
        """Test that pretrained model paths have correct format."""
        for name, path in constants.PRETRAINED_MODELS.items():
            assert path.startswith("hf://"), f"Model {name} path should start with hf://"


class TestVocoderConfig:
    """Tests for vocoder configuration."""

    def test_mel_spec_types(self):
        """Test that mel spec types list exists."""
        assert hasattr(constants, "MEL_SPEC_TYPES")
        assert "vocos" in constants.MEL_SPEC_TYPES
        assert "bigvgan" in constants.MEL_SPEC_TYPES

    def test_default_mel_spec_type(self):
        """Test default mel spec type."""
        assert constants.DEFAULT_MEL_SPEC_TYPE == "vocos"


class TestTokenizerConfig:
    """Tests for tokenizer configuration."""

    def test_tokenizer_types(self):
        """Test that tokenizer types list exists."""
        assert hasattr(constants, "TOKENIZER_TYPES")
        assert "pinyin" in constants.TOKENIZER_TYPES
        assert "char" in constants.TOKENIZER_TYPES
        assert "custom" in constants.TOKENIZER_TYPES

    def test_default_tokenizer(self):
        """Test default tokenizer type."""
        assert constants.DEFAULT_TOKENIZER == "pinyin"
