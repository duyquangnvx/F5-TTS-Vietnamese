"""
Utility functions for F5-TTS Finetune Gradio UI.

This module contains settings management and project utilities.
"""

from __future__ import annotations

import json
import os
from importlib.resources import files
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr


# Path configurations
path_data = str(files("f5_tts").joinpath("../../data"))
path_project_ckpts = str(files("f5_tts").joinpath("../../ckpts"))
file_train = str(files("f5_tts").joinpath("train/finetune_cli.py"))


def save_settings(
    project_name: str,
    exp_name: str,
    learning_rate: float,
    batch_size_per_gpu: int,
    batch_size_type: str,
    max_samples: int,
    grad_accumulation_steps: int,
    max_grad_norm: float,
    epochs: int,
    num_warmup_updates: int,
    save_per_updates: int,
    keep_last_n_checkpoints: int,
    last_per_updates: int,
    finetune: bool,
    file_checkpoint_train: str,
    tokenizer_type: str,
    tokenizer_file: str,
    mixed_precision: str,
    logger: str,
    ch_8bit_adam: bool,
) -> str:
    """Save training settings to a JSON file."""
    path_project = os.path.join(path_project_ckpts, project_name)
    os.makedirs(path_project, exist_ok=True)
    file_setting = os.path.join(path_project, "setting.json")

    settings = {
        "exp_name": exp_name,
        "learning_rate": learning_rate,
        "batch_size_per_gpu": batch_size_per_gpu,
        "batch_size_type": batch_size_type,
        "max_samples": max_samples,
        "grad_accumulation_steps": grad_accumulation_steps,
        "max_grad_norm": max_grad_norm,
        "epochs": epochs,
        "num_warmup_updates": num_warmup_updates,
        "save_per_updates": save_per_updates,
        "keep_last_n_checkpoints": keep_last_n_checkpoints,
        "last_per_updates": last_per_updates,
        "finetune": finetune,
        "file_checkpoint_train": file_checkpoint_train,
        "tokenizer_type": tokenizer_type,
        "tokenizer_file": tokenizer_file,
        "mixed_precision": mixed_precision,
        "logger": logger,
        "bnb_optimizer": ch_8bit_adam,
    }
    with open(file_setting, "w") as f:
        json.dump(settings, f, indent=4)
    return "Settings saved!"


def load_settings(project_name: str) -> Tuple:
    """Load training settings from a JSON file."""
    project_name = project_name.replace("_pinyin", "").replace("_char", "")
    path_project = os.path.join(path_project_ckpts, project_name)
    file_setting = os.path.join(path_project, "setting.json")

    # Default settings
    default_settings = {
        "exp_name": "F5TTS_v1_Base",
        "learning_rate": 1e-5,
        "batch_size_per_gpu": 1,
        "batch_size_type": "sample",
        "max_samples": 64,
        "grad_accumulation_steps": 4,
        "max_grad_norm": 1,
        "epochs": 100,
        "num_warmup_updates": 100,
        "save_per_updates": 500,
        "keep_last_n_checkpoints": -1,
        "last_per_updates": 100,
        "finetune": True,
        "file_checkpoint_train": "",
        "tokenizer_type": "pinyin",
        "tokenizer_file": "",
        "mixed_precision": "none",
        "logger": "wandb",
        "bnb_optimizer": False,
    }

    # Load settings from file if it exists
    if os.path.isfile(file_setting):
        with open(file_setting, "r") as f:
            file_settings = json.load(f)
        default_settings.update(file_settings)

    # Return as a tuple in the correct order
    return (
        default_settings["exp_name"],
        default_settings["learning_rate"],
        default_settings["batch_size_per_gpu"],
        default_settings["batch_size_type"],
        default_settings["max_samples"],
        default_settings["grad_accumulation_steps"],
        default_settings["max_grad_norm"],
        default_settings["epochs"],
        default_settings["num_warmup_updates"],
        default_settings["save_per_updates"],
        default_settings["keep_last_n_checkpoints"],
        default_settings["last_per_updates"],
        default_settings["finetune"],
        default_settings["file_checkpoint_train"],
        default_settings["tokenizer_type"],
        default_settings["tokenizer_file"],
        default_settings["mixed_precision"],
        default_settings["logger"],
        default_settings["bnb_optimizer"],
    )


def get_list_projects() -> Tuple[List[str], Optional[str]]:
    """Get list of available projects."""
    project_list = []
    if os.path.isdir(path_data):
        for folder in os.listdir(path_data):
            path_folder = os.path.join(path_data, folder)
            if not os.path.isdir(path_folder):
                continue
            folder = folder.lower()
            if folder == "emilia_zh_en_pinyin":
                continue
            project_list.append(folder)

    projects_select = None if not project_list else project_list[-1]
    return project_list, projects_select


def create_data_project(name: str, tokenizer_type: str):
    """Create a new data project directory."""
    name += "_" + tokenizer_type
    os.makedirs(os.path.join(path_data, name), exist_ok=True)
    os.makedirs(os.path.join(path_data, name, "dataset"), exist_ok=True)
    project_list, projects_select = get_list_projects()
    return gr.update(choices=project_list, value=name)


def format_seconds_to_hms(seconds: float) -> str:
    """Format seconds to HH:MM:SS string."""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = seconds % 60
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, int(seconds))


def get_correct_audio_path(
    audio_input: str,
    base_path: str = "wavs",
    supported_formats: Tuple[str, ...] = (
        "wav", "mp3", "aac", "flac", "m4a", "alac", "ogg", "aiff", "wma", "amr"
    ),
) -> Optional[str]:
    """Get the correct audio file path with extension detection."""
    file_audio = None

    def has_supported_extension(file_name: str) -> bool:
        return any(file_name.endswith(f".{ext}") for ext in supported_formats)

    # Case 1: If it's a full path with a valid extension, use it directly
    if os.path.isabs(audio_input) and has_supported_extension(audio_input):
        file_audio = audio_input

    # Case 2: If it has a supported extension but is not a full path
    elif has_supported_extension(audio_input) and not os.path.isabs(audio_input):
        file_audio = os.path.join(base_path, audio_input)

    # Case 3: If only the name is given (no extension and not a full path)
    elif not has_supported_extension(audio_input) and not os.path.isabs(audio_input):
        for ext in supported_formats:
            potential_file = os.path.join(base_path, f"{audio_input}.{ext}")
            if os.path.exists(potential_file):
                file_audio = potential_file
                break
        else:
            file_audio = os.path.join(base_path, f"{audio_input}.{supported_formats[0]}")

    return file_audio


def check_user(value: bool):
    """Toggle visibility for user mode checkbox."""
    return gr.update(visible=not value), gr.update(visible=value)


def check_finetune(finetune: bool):
    """Toggle interactivity based on finetune checkbox."""
    return (
        gr.update(interactive=finetune),
        gr.update(interactive=finetune),
        gr.update(interactive=finetune),
    )


def get_audio_select(file_sample: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Get audio reference and generated file paths from sample."""
    select_audio_ref = file_sample
    select_audio_gen = file_sample

    if file_sample is not None:
        select_audio_ref += "_ref.wav"
        select_audio_gen += "_gen.wav"

    return select_audio_ref, select_audio_gen
