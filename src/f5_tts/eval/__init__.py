"""
Evaluation utilities for F5-TTS.

This module provides utilities for evaluating F5-TTS models,
including metrics computation and benchmark evaluation.
"""

from f5_tts.eval.utils_eval import (
    get_seedtts_testset_metainfo,
    get_librispeech_test_clean_metainfo,
    get_inference_prompt,
    run_sim_metric,
    run_wer_metric,
)

__all__ = [
    "get_seedtts_testset_metainfo",
    "get_librispeech_test_clean_metainfo",
    "get_inference_prompt",
    "run_sim_metric",
    "run_wer_metric",
]
