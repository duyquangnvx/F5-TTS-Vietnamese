"""
Callback functions for F5-TTS Finetune Gradio UI.

This module contains all callback functions for data processing,
transcription, vocabulary management, and inference.
"""

from __future__ import annotations

import gc
import json
import os
import random
import shutil
import tempfile
from glob import glob
from typing import Any, List, Optional, Tuple

import gradio as gr
import librosa
import numpy as np
import psutil
import torch
from cached_path import cached_path
from datasets import Dataset as Dataset_
from datasets.arrow_writer import ArrowWriter
from safetensors.torch import load_file, save_file
from scipy.io import wavfile

from f5_tts.model.utils import convert_char_to_pinyin
from f5_tts.infer.utils_infer import transcribe
from f5_tts.train.finetune_gradio.audio_processing import (
    Slicer,
    get_audio_duration,
    clear_text,
)
from f5_tts.train.finetune_gradio.utils import (
    path_data,
    path_project_ckpts,
    get_correct_audio_path,
    format_seconds_to_hms,
)


# Module-level state for inference
tts_api = None
last_checkpoint = ""
last_device = ""
last_ema = None


def transcribe_all(
    name_project: str,
    audio_files: Optional[List[str]],
    language: str,
    user: bool = False,
    progress: gr.Progress = gr.Progress(),
) -> str:
    """Transcribe all audio files and create metadata."""
    path_project = os.path.join(path_data, name_project)
    path_dataset = os.path.join(path_project, "dataset")
    path_project_wavs = os.path.join(path_project, "wavs")
    file_metadata = os.path.join(path_project, "metadata.csv")

    if not user:
        if audio_files is None:
            return "You need to load an audio file."

    if os.path.isdir(path_project_wavs):
        shutil.rmtree(path_project_wavs)

    if os.path.isfile(file_metadata):
        os.remove(file_metadata)

    os.makedirs(path_project_wavs, exist_ok=True)

    if user:
        file_audios = [
            file
            for format in ("*.wav", "*.ogg", "*.opus", "*.mp3", "*.flac")
            for file in glob(os.path.join(path_dataset, format))
        ]
        if file_audios == []:
            return "No audio file was found in the dataset."
    else:
        file_audios = audio_files

    alpha = 0.5
    _max = 1.0
    slicer = Slicer(24000)

    num = 0
    error_num = 0
    data = ""
    for file_audio in progress.tqdm(file_audios, desc="transcribe files", total=len((file_audios))):
        audio, _ = librosa.load(file_audio, sr=24000, mono=True)

        list_slicer = slicer.slice(audio)
        for chunk, start, end in progress.tqdm(list_slicer, total=len(list_slicer), desc="slicer files"):
            name_segment = os.path.join(f"segment_{num}")
            file_segment = os.path.join(path_project_wavs, f"{name_segment}.wav")

            tmp_max = np.abs(chunk).max()
            if tmp_max > 1:
                chunk /= tmp_max
            chunk = (chunk / tmp_max * (_max * alpha)) + (1 - alpha) * chunk
            wavfile.write(file_segment, 24000, (chunk * 32767).astype(np.int16))

            try:
                text = transcribe(file_segment, language)
                text = text.lower().strip().replace('"', "")

                data += f"{name_segment}|{text}\n"

                num += 1
            except:  # noqa: E722
                error_num += 1

    with open(file_metadata, "w", encoding="utf-8-sig") as f:
        f.write(data)

    if error_num != []:
        error_text = f"\nerror files : {error_num}"
    else:
        error_text = ""

    return f"transcribe complete samples : {num}\npath : {path_project_wavs}{error_text}"


def create_metadata(
    name_project: str,
    ch_tokenizer: bool,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[str, str]:
    """Create metadata files (raw.arrow, duration.json, vocab.txt) from transcription."""
    path_project = os.path.join(path_data, name_project)
    path_project_wavs = os.path.join(path_project, "wavs")
    file_metadata = os.path.join(path_project, "metadata.csv")
    file_raw = os.path.join(path_project, "raw.arrow")
    file_duration = os.path.join(path_project, "duration.json")
    file_vocab = os.path.join(path_project, "vocab.txt")

    if not os.path.isfile(file_metadata):
        return "The file was not found in " + file_metadata, ""

    with open(file_metadata, "r", encoding="utf-8-sig") as f:
        data = f.read()

    audio_path_list = []
    text_list = []
    duration_list = []

    count = data.split("\n")
    lenght = 0
    result = []
    error_files = []
    text_vocab_set = set()

    for line in progress.tqdm(data.split("\n"), total=count):
        sp_line = line.split("|")
        if len(sp_line) != 2:
            continue
        name_audio, text = sp_line[:2]

        file_audio = get_correct_audio_path(name_audio, path_project_wavs)

        if not os.path.isfile(file_audio):
            error_files.append([file_audio, "error path"])
            continue

        try:
            duration = get_audio_duration(file_audio)
        except Exception as e:
            error_files.append([file_audio, "duration"])
            print(f"Error processing {file_audio}: {e}")
            continue

        if duration < 1 or duration > 30:
            if duration > 30:
                error_files.append([file_audio, "duration > 30 sec"])
            if duration < 1:
                error_files.append([file_audio, "duration < 1 sec "])
            continue
        if len(text) < 3:
            error_files.append([file_audio, "very short text length 3"])
            continue

        text = clear_text(text)
        text = convert_char_to_pinyin([text], polyphone=True)[0]

        audio_path_list.append(file_audio)
        duration_list.append(duration)
        text_list.append(text)

        result.append({"audio_path": file_audio, "text": text, "duration": duration})
        if ch_tokenizer:
            text_vocab_set.update(list(text))

        lenght += duration

    if duration_list == []:
        return f"Error: No audio files found in the specified path : {path_project_wavs}", ""

    min_second = round(min(duration_list), 2)
    max_second = round(max(duration_list), 2)

    with ArrowWriter(path=file_raw, writer_batch_size=1) as writer:
        for line in progress.tqdm(result, total=len(result), desc="prepare data"):
            writer.write(line)

    with open(file_duration, "w") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    new_vocal = ""
    if not ch_tokenizer:
        if not os.path.isfile(file_vocab):
            file_vocab_finetune = os.path.join(path_data, "Emilia_ZH_EN_pinyin/vocab.txt")
            if not os.path.isfile(file_vocab_finetune):
                return "Error: Vocabulary file 'Emilia_ZH_EN_pinyin' not found!", ""
            shutil.copy2(file_vocab_finetune, file_vocab)

        with open(file_vocab, "r", encoding="utf-8-sig") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    else:
        with open(file_vocab, "w", encoding="utf-8-sig") as f:
            for vocab in sorted(text_vocab_set):
                f.write(vocab + "\n")
                new_vocal += vocab + "\n"
        vocab_size = len(text_vocab_set)

    if error_files != []:
        error_text = "\n".join([" = ".join(item) for item in error_files])
    else:
        error_text = ""

    return (
        f"prepare complete \nsamples : {len(text_list)}\ntime data : {format_seconds_to_hms(lenght)}\nmin sec : {min_second}\nmax sec : {max_second}\nfile_arrow : {file_raw}\nvocab : {vocab_size}\n{error_text}",
        new_vocal,
    )


def vocab_check(project_name: str) -> Tuple[str, str]:
    """Check vocabulary against the base Emilia vocabulary."""
    name_project = project_name
    path_project = os.path.join(path_data, name_project)

    file_metadata = os.path.join(path_project, "metadata.csv")

    file_vocab = os.path.join(path_data, "Emilia_ZH_EN_pinyin/vocab.txt")
    if not os.path.isfile(file_vocab):
        return f"the file {file_vocab} not found !", ""

    with open(file_vocab, "r", encoding="utf-8-sig") as f:
        data = f.read()
        vocab = data.split("\n")
        vocab = set(vocab)

    if not os.path.isfile(file_metadata):
        return f"the file {file_metadata} not found !", ""

    with open(file_metadata, "r", encoding="utf-8-sig") as f:
        data = f.read()

    miss_symbols = []
    miss_symbols_keep = {}
    for item in data.split("\n"):
        sp = item.split("|")
        if len(sp) != 2:
            continue

        text = sp[1].lower().strip()

        for t in text:
            if t not in vocab and t not in miss_symbols_keep:
                miss_symbols.append(t)
                miss_symbols_keep[t] = t

    if miss_symbols == []:
        vocab_miss = ""
        info = "You can train using your language !"
    else:
        vocab_miss = ",".join(miss_symbols)
        info = f"The following symbols are missing in your language {len(miss_symbols)}\n\n"

    return info, vocab_miss


def vocab_count(text: str) -> str:
    """Count the number of vocabulary items."""
    return str(len(text.split(",")))


def expand_model_embeddings(
    ckpt_path: str,
    new_ckpt_path: str,
    num_new_tokens: int = 42,
) -> int:
    """Expand model embeddings to accommodate new vocabulary tokens."""
    seed = 666
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if ckpt_path.endswith(".safetensors"):
        ckpt = load_file(ckpt_path, device="cpu")
        ckpt = {"ema_model_state_dict": ckpt}
    elif ckpt_path.endswith(".pt"):
        ckpt = torch.load(ckpt_path, map_location="cpu")

    ema_sd = ckpt.get("ema_model_state_dict", {})
    embed_key_ema = "ema_model.transformer.text_embed.text_embed.weight"
    old_embed_ema = ema_sd[embed_key_ema]

    vocab_old = old_embed_ema.size(0)
    embed_dim = old_embed_ema.size(1)
    vocab_new = vocab_old + num_new_tokens

    def expand_embeddings(old_embeddings):
        new_embeddings = torch.zeros((vocab_new, embed_dim))
        new_embeddings[:vocab_old] = old_embeddings
        new_embeddings[vocab_old:] = torch.randn((num_new_tokens, embed_dim))
        return new_embeddings

    ema_sd[embed_key_ema] = expand_embeddings(ema_sd[embed_key_ema])

    torch.save(ckpt, new_ckpt_path)

    return vocab_new


def vocab_extend(project_name: str, symbols: str, model_type: str) -> str:
    """Extend vocabulary with new symbols and create expanded model."""
    if symbols == "":
        return "Symbols empty!"

    name_project = project_name
    path_project = os.path.join(path_data, name_project)
    file_vocab_project = os.path.join(path_project, "vocab.txt")

    file_vocab = os.path.join(path_data, "Emilia_ZH_EN_pinyin/vocab.txt")
    if not os.path.isfile(file_vocab):
        return f"the file {file_vocab} not found !"

    symbols = symbols.split(",")
    if symbols == []:
        return "Symbols to extend not found."

    with open(file_vocab, "r", encoding="utf-8-sig") as f:
        data = f.read()
        vocab = data.split("\n")
    vocab_check = set(vocab)

    miss_symbols = []
    for item in symbols:
        item = item.replace(" ", "")
        if item in vocab_check:
            continue
        miss_symbols.append(item)

    if miss_symbols == []:
        return "Symbols are okay no need to extend."

    size_vocab = len(vocab)
    vocab.pop()
    for item in miss_symbols:
        vocab.append(item)

    vocab.append("")

    with open(file_vocab_project, "w", encoding="utf-8") as f:
        f.write("\n".join(vocab))

    if model_type == "F5TTS_v1_Base":
        ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"))
    elif model_type == "F5TTS_Base":
        ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt"))
    elif model_type == "E2TTS_Base":
        ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.pt"))

    vocab_size_new = len(miss_symbols)

    dataset_name = name_project.replace("_pinyin", "").replace("_char", "")
    new_ckpt_path = os.path.join(path_project_ckpts, dataset_name)
    os.makedirs(new_ckpt_path, exist_ok=True)

    # Add pretrained_ prefix to model when copying for consistency with finetune_cli.py
    new_ckpt_file = os.path.join(new_ckpt_path, "pretrained_" + os.path.basename(ckpt_path))

    size = expand_model_embeddings(ckpt_path, new_ckpt_file, num_new_tokens=vocab_size_new)

    vocab_new = "\n".join(miss_symbols)
    return f"vocab old size : {size_vocab}\nvocab new size : {size}\nvocab add : {vocab_size_new}\nnew symbols :\n{vocab_new}"


def extract_and_save_ema_model(
    checkpoint_path: str,
    new_checkpoint_path: str,
    safetensors: bool,
) -> str:
    """Extract EMA model from checkpoint and save it separately."""
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        print("Original Checkpoint Keys:", checkpoint.keys())

        ema_model_state_dict = checkpoint.get("ema_model_state_dict", None)
        if ema_model_state_dict is None:
            return "No 'ema_model_state_dict' found in the checkpoint."

        if safetensors:
            new_checkpoint_path = new_checkpoint_path.replace(".pt", ".safetensors")
            save_file(ema_model_state_dict, new_checkpoint_path)
        else:
            new_checkpoint_path = new_checkpoint_path.replace(".safetensors", ".pt")
            new_checkpoint = {"ema_model_state_dict": ema_model_state_dict}
            torch.save(new_checkpoint, new_checkpoint_path)

        return f"New checkpoint saved at: {new_checkpoint_path}"

    except Exception as e:
        return f"An error occurred: {e}"


def get_random_sample_prepare(project_name: str) -> Tuple[str, Optional[str]]:
    """Get a random sample from prepared data for preview."""
    name_project = project_name
    path_project = os.path.join(path_data, name_project)
    file_arrow = os.path.join(path_project, "raw.arrow")
    if not os.path.isfile(file_arrow):
        return "", None
    dataset = Dataset_.from_file(file_arrow)
    random_sample = dataset.shuffle(seed=random.randint(0, 1000)).select([0])
    text = "[" + " , ".join(["' " + t + " '" for t in random_sample["text"][0]]) + "]"
    audio_path = random_sample["audio_path"][0]
    return text, audio_path


def get_random_sample_transcribe(project_name: str) -> Tuple[str, Optional[str]]:
    """Get a random sample from transcribed data for preview."""
    name_project = project_name
    path_project = os.path.join(path_data, name_project)
    file_metadata = os.path.join(path_project, "metadata.csv")
    if not os.path.isfile(file_metadata):
        return "", None

    data = ""
    with open(file_metadata, "r", encoding="utf-8-sig") as f:
        data = f.read()

    list_data = []
    for item in data.split("\n"):
        sp = item.split("|")
        if len(sp) != 2:
            continue

        file_audio = get_correct_audio_path(sp[0], os.path.join(path_project, "wavs"))
        list_data.append([file_audio, sp[1]])

    if list_data == []:
        return "", None

    random_item = random.choice(list_data)

    return random_item[1], random_item[0]


def get_random_sample_infer(project_name: str) -> Tuple[str, str, Optional[str]]:
    """Get a random sample for inference testing."""
    text, audio = get_random_sample_transcribe(project_name)
    return (
        text,
        text,
        audio,
    )


def infer(
    project: str,
    file_checkpoint: str,
    exp_name: str,
    ref_text: str,
    ref_audio: str,
    gen_text: str,
    nfe_step: int,
    use_ema: bool,
    speed: float,
    seed: int,
    remove_silence: bool,
) -> Tuple[Optional[str], str, str]:
    """Run inference with the specified model and parameters."""
    global last_checkpoint, last_device, tts_api, last_ema

    from f5_tts.api import F5TTS
    from f5_tts.train.finetune_gradio.training_manager import training_process

    if not os.path.isfile(file_checkpoint):
        return None, "checkpoint not found!"

    if training_process is not None:
        device_test = "cpu"
    else:
        device_test = None

    if last_checkpoint != file_checkpoint or last_device != device_test or last_ema != use_ema or tts_api is None:
        if last_checkpoint != file_checkpoint:
            last_checkpoint = file_checkpoint

        if last_device != device_test:
            last_device = device_test

        if last_ema != use_ema:
            last_ema = use_ema

        vocab_file = os.path.join(path_data, project, "vocab.txt")

        tts_api = F5TTS(
            model=exp_name, ckpt_file=file_checkpoint, vocab_file=vocab_file, device=device_test, use_ema=use_ema
        )

        print("update >> ", device_test, file_checkpoint, use_ema)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        tts_api.infer(
            ref_file=ref_audio,
            ref_text=ref_text.lower().strip(),
            gen_text=gen_text.lower().strip(),
            nfe_step=nfe_step,
            speed=speed,
            remove_silence=remove_silence,
            file_wave=f.name,
            seed=seed,
        )
        return f.name, tts_api.device, str(tts_api.seed)


def get_checkpoints_project(
    project_name: Optional[str],
    is_gradio: bool = True,
) -> Any:
    """Get list of checkpoints for a project."""
    if project_name is None:
        return [], ""
    project_name = project_name.replace("_pinyin", "").replace("_char", "")

    if os.path.isdir(path_project_ckpts):
        files_checkpoints = glob(os.path.join(path_project_ckpts, project_name, "*.pt"))
        # Separate pretrained and regular checkpoints
        pretrained_checkpoints = [f for f in files_checkpoints if "pretrained_" in os.path.basename(f)]
        regular_checkpoints = [
            f
            for f in files_checkpoints
            if "pretrained_" not in os.path.basename(f) and "model_last.pt" not in os.path.basename(f)
        ]
        last_checkpoint = [f for f in files_checkpoints if "model_last.pt" in os.path.basename(f)]

        # Sort regular checkpoints by number
        regular_checkpoints = sorted(
            regular_checkpoints, key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
        )

        # Combine in order: pretrained, regular, last
        files_checkpoints = pretrained_checkpoints + regular_checkpoints + last_checkpoint
    else:
        files_checkpoints = []

    selelect_checkpoint = None if not files_checkpoints else files_checkpoints[0]

    if is_gradio:
        return gr.update(choices=files_checkpoints, value=selelect_checkpoint)

    return files_checkpoints, selelect_checkpoint


def get_audio_project(
    project_name: Optional[str],
    is_gradio: bool = True,
) -> Any:
    """Get list of audio samples for a project."""
    if project_name is None:
        return [], ""
    project_name = project_name.replace("_pinyin", "").replace("_char", "")

    if os.path.isdir(path_project_ckpts):
        files_audios = glob(os.path.join(path_project_ckpts, project_name, "samples", "*.wav"))
        files_audios = sorted(files_audios, key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))

        files_audios = [item.replace("_gen.wav", "") for item in files_audios if item.endswith("_gen.wav")]
    else:
        files_audios = []

    selelect_checkpoint = None if not files_audios else files_audios[0]

    if is_gradio:
        return gr.update(choices=files_audios, value=selelect_checkpoint)

    return files_audios, selelect_checkpoint


def get_gpu_stats() -> str:
    """Get GPU statistics."""
    gpu_stats = ""

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_properties = torch.cuda.get_device_properties(i)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(i) / (1024**2)
            reserved_memory = torch.cuda.memory_reserved(i) / (1024**2)

            gpu_stats += (
                f"GPU {i} Name: {gpu_name}\n"
                f"Total GPU memory (GPU {i}): {total_memory:.2f} GB\n"
                f"Allocated GPU memory (GPU {i}): {allocated_memory:.2f} MB\n"
                f"Reserved GPU memory (GPU {i}): {reserved_memory:.2f} MB\n\n"
            )
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        gpu_count = torch.xpu.device_count()
        for i in range(gpu_count):
            gpu_name = torch.xpu.get_device_name(i)
            gpu_properties = torch.xpu.get_device_properties(i)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.xpu.memory_allocated(i) / (1024**2)
            reserved_memory = torch.xpu.memory_reserved(i) / (1024**2)

            gpu_stats += (
                f"GPU {i} Name: {gpu_name}\n"
                f"Total GPU memory (GPU {i}): {total_memory:.2f} GB\n"
                f"Allocated GPU memory (GPU {i}): {allocated_memory:.2f} MB\n"
                f"Reserved GPU memory (GPU {i}): {reserved_memory:.2f} MB\n\n"
            )
    elif torch.backends.mps.is_available():
        gpu_count = 1
        gpu_stats += "MPS GPU\n"
        total_memory = psutil.virtual_memory().total / (1024**3)
        allocated_memory = 0
        reserved_memory = 0

        gpu_stats += (
            f"Total system memory: {total_memory:.2f} GB\n"
            f"Allocated GPU memory (MPS): {allocated_memory:.2f} MB\n"
            f"Reserved GPU memory (MPS): {reserved_memory:.2f} MB\n"
        )

    else:
        gpu_stats = "No GPU available"

    return gpu_stats


def get_cpu_stats() -> str:
    """Get CPU statistics."""
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    memory_used = memory_info.used / (1024**2)
    memory_total = memory_info.total / (1024**2)
    memory_percent = memory_info.percent

    pid = os.getpid()
    process = psutil.Process(pid)
    nice_value = process.nice()

    cpu_stats = (
        f"CPU Usage: {cpu_usage:.2f}%\n"
        f"System Memory: {memory_used:.2f} MB used / {memory_total:.2f} MB total ({memory_percent}% used)\n"
        f"Process Priority (Nice value): {nice_value}"
    )

    return cpu_stats


def get_combined_stats() -> str:
    """Get combined GPU and CPU statistics."""
    gpu_stats = get_gpu_stats()
    cpu_stats = get_cpu_stats()
    combined_stats = f"### GPU Stats\n{gpu_stats}\n\n### CPU Stats\n{cpu_stats}"
    return combined_stats
