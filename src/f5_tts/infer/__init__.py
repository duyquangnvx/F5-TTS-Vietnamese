"""
Inference utilities for F5-TTS.

This module provides utilities for running inference with F5-TTS models,
including model loading, text processing, and audio generation.
"""

from f5_tts.infer.utils_infer import (
    # Context-based API (recommended)
    InferenceContext,
    get_default_context,
    # Model loading
    load_model,
    load_vocoder,
    load_checkpoint,
    # Transcription
    transcribe,
    initialize_asr_pipeline,
    # Preprocessing
    preprocess_ref_audio_text,
    chunk_text,
    # Inference
    infer_process,
    infer_batch_process,
    # Post-processing
    remove_silence_for_generated_wav,
    remove_silence_edges,
    save_spectrogram,
)

__all__ = [
    # Context-based API (recommended)
    "InferenceContext",
    "get_default_context",
    # Model loading
    "load_model",
    "load_vocoder",
    "load_checkpoint",
    # Transcription
    "transcribe",
    "initialize_asr_pipeline",
    # Preprocessing
    "preprocess_ref_audio_text",
    "chunk_text",
    # Inference
    "infer_process",
    "infer_batch_process",
    # Post-processing
    "remove_silence_for_generated_wav",
    "remove_silence_edges",
    "save_spectrogram",
]
