#!/usr/bin/env python3
"""
Pre-download models for offline use.

This script downloads required models from HuggingFace so they are cached
and available for offline inference.

Usage:
    python -m f5_tts.tools.download_models
    python -m f5_tts.tools.download_models --vocoder vocos
    python -m f5_tts.tools.download_models --all
"""

import argparse
import os
import sys

# Fix UTF-8 encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def download_vocos():
    """Download Vocos vocoder."""
    from huggingface_hub import hf_hub_download

    repo_id = "charactr/vocos-mel-24khz"
    print(f"Downloading Vocos vocoder from: {repo_id}")

    files = ["config.yaml", "pytorch_model.bin"]
    for filename in files:
        print(f"  - {filename}...", end=" ", flush=True)
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"OK ({path})")

    print("Vocos vocoder downloaded successfully!")


def download_bigvgan():
    """Download BigVGAN vocoder."""
    from huggingface_hub import snapshot_download

    repo_id = "nvidia/bigvgan_v2_24khz_100band_256x"
    print(f"Downloading BigVGAN vocoder from: {repo_id}")

    path = snapshot_download(repo_id=repo_id)
    print(f"BigVGAN vocoder downloaded successfully to: {path}")


def download_f5tts_base():
    """Download F5-TTS Base model."""
    from cached_path import cached_path

    models = [
        ("F5-TTS Base (vocos)", "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"),
        ("F5-TTS v1 Base", "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"),
    ]

    for name, url in models:
        print(f"Downloading {name}...")
        try:
            path = cached_path(url)
            print(f"  Downloaded to: {path}")
        except Exception as e:
            print(f"  Failed: {e}")


def download_whisper():
    """Download Whisper model for ASR."""
    print("Downloading Whisper model for automatic speech recognition...")
    try:
        from transformers import pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            torch_dtype="float16",
            device="cpu",  # Just download, don't load to GPU
        )
        print("Whisper model downloaded successfully!")
        del pipe
    except Exception as e:
        print(f"Failed to download Whisper: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download models for F5-TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m f5_tts.tools.download_models              # Download vocoder only
    python -m f5_tts.tools.download_models --vocoder vocos
    python -m f5_tts.tools.download_models --vocoder bigvgan
    python -m f5_tts.tools.download_models --f5tts      # Download F5-TTS models
    python -m f5_tts.tools.download_models --whisper    # Download Whisper ASR
    python -m f5_tts.tools.download_models --all        # Download everything
        """,
    )
    parser.add_argument(
        "--vocoder",
        type=str,
        choices=["vocos", "bigvgan", "all"],
        default="vocos",
        help="Vocoder to download (default: vocos)",
    )
    parser.add_argument(
        "--f5tts",
        action="store_true",
        help="Download F5-TTS pretrained models",
    )
    parser.add_argument(
        "--whisper",
        action="store_true",
        help="Download Whisper ASR model",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all models",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("F5-TTS Model Downloader")
    print("=" * 50)
    print()

    if args.all:
        download_vocos()
        print()
        download_bigvgan()
        print()
        download_f5tts_base()
        print()
        download_whisper()
    else:
        # Download vocoder
        if args.vocoder in ["vocos", "all"]:
            download_vocos()
            print()
        if args.vocoder in ["bigvgan", "all"]:
            download_bigvgan()
            print()

        # Download F5-TTS models
        if args.f5tts:
            download_f5tts_base()
            print()

        # Download Whisper
        if args.whisper:
            download_whisper()
            print()

    print("=" * 50)
    print("Download complete!")
    print("Models are cached in ~/.cache/huggingface/hub/")
    print("=" * 50)


if __name__ == "__main__":
    main()
