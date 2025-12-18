# A unified script for inference process
# Make adjustments inside functions, and consider both gradio and cli scripts if need to change func output format

from __future__ import annotations

import hashlib
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from typing import TYPE_CHECKING, Callable, Dict, Generator, List, Optional, Tuple, Union

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # for MPS device compatibility
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../third_party/BigVGAN/")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pylab as plt
import numpy as np
import torch
import torchaudio
import tqdm
from huggingface_hub import hf_hub_download, snapshot_download
from pydub import AudioSegment, silence
from transformers import pipeline
from vocos import Vocos

from f5_tts.model import CFM
from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer
from f5_tts.constants import (
    TARGET_SAMPLE_RATE,
    N_MEL_CHANNELS,
    HOP_LENGTH,
    WIN_LENGTH,
    N_FFT,
    DEFAULT_NFE_STEP,
    DEFAULT_CFG_STRENGTH,
    DEFAULT_SPEED,
    DEFAULT_TARGET_RMS,
    DEFAULT_CROSS_FADE_DURATION,
    DEFAULT_SWAY_SAMPLING_COEF,
)
from f5_tts.core.device import get_device, get_dtype

if TYPE_CHECKING:
    from torch import Tensor


# -----------------------------------------
# Module-level defaults (for backward compatibility)
# -----------------------------------------

target_sample_rate = TARGET_SAMPLE_RATE
n_mel_channels = N_MEL_CHANNELS
hop_length = HOP_LENGTH
win_length = WIN_LENGTH
n_fft = N_FFT
mel_spec_type = "vocos"
target_rms = DEFAULT_TARGET_RMS
cross_fade_duration = DEFAULT_CROSS_FADE_DURATION
ode_method = "euler"
nfe_step = DEFAULT_NFE_STEP
cfg_strength = DEFAULT_CFG_STRENGTH
sway_sampling_coef = DEFAULT_SWAY_SAMPLING_COEF
speed = DEFAULT_SPEED
fix_duration = None


# -----------------------------------------
# InferenceContext class
# -----------------------------------------

class InferenceContext:
    """
    Encapsulates inference state and configuration.

    This class replaces global state with instance-based state management,
    making it easier to manage multiple inference sessions and test code.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        n_mel_channels: int = N_MEL_CHANNELS,
        hop_length: int = HOP_LENGTH,
        win_length: int = WIN_LENGTH,
        n_fft: int = N_FFT,
        mel_spec_type: str = "vocos",
        target_rms: float = DEFAULT_TARGET_RMS,
        cross_fade_duration: float = DEFAULT_CROSS_FADE_DURATION,
        ode_method: str = "euler",
        nfe_step: int = DEFAULT_NFE_STEP,
        cfg_strength: float = DEFAULT_CFG_STRENGTH,
        sway_sampling_coef: float = DEFAULT_SWAY_SAMPLING_COEF,
        speed: float = DEFAULT_SPEED,
        fix_duration: Optional[float] = None,
    ):
        """Initialize inference context with configuration."""
        self.device = device or get_device()
        self.target_sample_rate = target_sample_rate
        self.n_mel_channels = n_mel_channels
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.mel_spec_type = mel_spec_type
        self.target_rms = target_rms
        self.cross_fade_duration = cross_fade_duration
        self.ode_method = ode_method
        self.nfe_step = nfe_step
        self.cfg_strength = cfg_strength
        self.sway_sampling_coef = sway_sampling_coef
        self.speed = speed
        self.fix_duration = fix_duration

        # Internal state
        self._ref_audio_cache: Dict[str, str] = {}
        self._asr_pipe = None
        self._vocoder = None
        self._model = None

    def clear_cache(self) -> None:
        """Clear the reference audio transcription cache."""
        self._ref_audio_cache.clear()

    def initialize_asr_pipeline(self, dtype: Optional[torch.dtype] = None) -> None:
        """Initialize the ASR pipeline for transcription."""
        if dtype is None:
            dtype = get_dtype(self.device)

        self._asr_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            torch_dtype=dtype,
            device=self.device,
        )

    def transcribe(self, ref_audio: str, language: Optional[str] = None) -> str:
        """Transcribe audio using the ASR pipeline."""
        if self._asr_pipe is None:
            self.initialize_asr_pipeline()

        return self._asr_pipe(
            ref_audio,
            chunk_length_s=30,
            batch_size=128,
            generate_kwargs={"task": "transcribe", "language": language} if language else {"task": "transcribe"},
            return_timestamps=False,
        )["text"].strip()

    def load_vocoder(
        self,
        vocoder_name: str = "vocos",
        is_local: bool = False,
        local_path: str = "",
        hf_cache_dir: Optional[str] = None,
    ) -> torch.nn.Module:
        """Load the vocoder model."""
        self._vocoder = load_vocoder(
            vocoder_name=vocoder_name,
            is_local=is_local,
            local_path=local_path,
            device=self.device,
            hf_cache_dir=hf_cache_dir,
        )
        return self._vocoder

    def load_model(
        self,
        model_cls,
        model_cfg: dict,
        ckpt_path: str,
        vocab_file: str = "",
        use_ema: bool = True,
    ) -> CFM:
        """Load the TTS model."""
        self._model = load_model(
            model_cls=model_cls,
            model_cfg=model_cfg,
            ckpt_path=ckpt_path,
            mel_spec_type=self.mel_spec_type,
            vocab_file=vocab_file,
            ode_method=self.ode_method,
            use_ema=use_ema,
            device=self.device,
        )
        return self._model

    def preprocess_ref_audio_text(
        self,
        ref_audio_orig: str,
        ref_text: str,
        clip_short: bool = True,
        show_info: Callable = print,
    ) -> Tuple[str, str]:
        """Preprocess reference audio and text."""
        return preprocess_ref_audio_text(
            ref_audio_orig=ref_audio_orig,
            ref_text=ref_text,
            clip_short=clip_short,
            show_info=show_info,
            device=self.device,
            ref_audio_cache=self._ref_audio_cache,
            transcribe_fn=self.transcribe,
        )

    @property
    def vocoder(self) -> Optional[torch.nn.Module]:
        """Get the loaded vocoder."""
        return self._vocoder

    @property
    def model(self) -> Optional[CFM]:
        """Get the loaded model."""
        return self._model


# -----------------------------------------
# Default context for backward compatibility
# -----------------------------------------

_default_context: Optional[InferenceContext] = None

def get_default_context() -> InferenceContext:
    """Get or create the default inference context."""
    global _default_context
    if _default_context is None:
        _default_context = InferenceContext()
    return _default_context


# Keep backward-compatible global variables
device = get_device()
_ref_audio_cache: Dict[str, str] = {}
asr_pipe = None


# -----------------------------------------
# Text processing utilities
# -----------------------------------------

def chunk_text(text: str, max_chars: int = 135) -> List[str]:
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text: The text to be split.
        max_chars: The maximum number of characters per chunk.

    Returns:
        A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# -----------------------------------------
# Model loading utilities
# -----------------------------------------

def load_vocoder(
    vocoder_name: str = "vocos",
    is_local: bool = False,
    local_path: str = "",
    device: str = device,
    hf_cache_dir: Optional[str] = None,
) -> torch.nn.Module:
    """Load vocoder model for audio generation."""
    if vocoder_name == "vocos":
        if is_local:
            print(f"Load vocos from local path {local_path}")
            config_path = f"{local_path}/config.yaml"
            model_path = f"{local_path}/pytorch_model.bin"
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            repo_id = "charactr/vocos-mel-24khz"
            config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
            model_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")
        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        from vocos.feature_extractors import EncodecFeatures

        if isinstance(vocoder.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in vocoder.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        vocoder.load_state_dict(state_dict)
        vocoder = vocoder.eval().to(device)
    elif vocoder_name == "bigvgan":
        try:
            from third_party.BigVGAN import bigvgan
        except ImportError:
            print("You need to follow the README to init submodule and change the BigVGAN source code.")
        if is_local:
            """download from https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main"""
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        else:
            local_path = snapshot_download(repo_id="nvidia/bigvgan_v2_24khz_100band_256x", cache_dir=hf_cache_dir)
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)

        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    return vocoder


def initialize_asr_pipeline(device: str = device, dtype: Optional[torch.dtype] = None) -> None:
    """Initialize the ASR pipeline for transcription (backward compatible)."""
    if dtype is None:
        dtype = get_dtype(device)

    global asr_pipe
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=dtype,
        device=device,
    )


def transcribe(ref_audio: str, language: Optional[str] = None) -> str:
    """Transcribe audio using the ASR pipeline (backward compatible)."""
    global asr_pipe
    if asr_pipe is None:
        initialize_asr_pipeline(device=device)
    return asr_pipe(
        ref_audio,
        chunk_length_s=30,
        batch_size=128,
        generate_kwargs={"task": "transcribe", "language": language} if language else {"task": "transcribe"},
        return_timestamps=False,
    )["text"].strip()


def load_checkpoint(
    model: CFM,
    ckpt_path: str,
    device: str,
    dtype: Optional[torch.dtype] = None,
    use_ema: bool = True,
) -> CFM:
    """Load model checkpoint for inference."""
    if dtype is None:
        dtype = get_dtype(device)
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file
        checkpoint = load_file(ckpt_path, device=device)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

        # patch for backward compatibility, 305e3ea
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    torch.cuda.empty_cache()

    return model.to(device)


def load_model(
    model_cls,
    model_cfg: dict,
    ckpt_path: str,
    mel_spec_type: str = mel_spec_type,
    vocab_file: str = "",
    ode_method: str = ode_method,
    use_ema: bool = True,
    device: str = device,
) -> CFM:
    """Load model for inference."""
    if vocab_file == "":
        vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
    tokenizer = "custom"

    print("\nvocab : ", vocab_file)
    print("token : ", tokenizer)
    print("model : ", ckpt_path, "\n")

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    return model


# -----------------------------------------
# Audio processing utilities
# -----------------------------------------

def remove_silence_edges(audio: AudioSegment, silence_threshold: int = -42) -> AudioSegment:
    """Remove silence from the edges of an audio segment."""
    # Remove silence from the start
    non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[non_silent_start_idx:]

    # Remove silence from the end
    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

    return trimmed_audio


def preprocess_ref_audio_text(
    ref_audio_orig: str,
    ref_text: str,
    clip_short: bool = True,
    show_info: Callable = print,
    device: str = device,
    ref_audio_cache: Optional[Dict[str, str]] = None,
    transcribe_fn: Optional[Callable] = None,
) -> Tuple[str, str]:
    """Preprocess reference audio and text for inference."""
    # Use global cache if not provided
    if ref_audio_cache is None:
        global _ref_audio_cache
        ref_audio_cache = _ref_audio_cache

    # Use global transcribe if not provided
    if transcribe_fn is None:
        transcribe_fn = transcribe

    show_info("Converting audio...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_orig)

        if clip_short:
            # 1. try to find long silence for clipping
            non_silent_segs = silence.split_on_silence(
                aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
                    show_info("Audio is over 15s, clipping short. (1)")
                    break
                non_silent_wave += non_silent_seg

            # 2. try to find short silence for clipping if 1. failed
            if len(non_silent_wave) > 12000:
                non_silent_segs = silence.split_on_silence(
                    aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
                        show_info("Audio is over 15s, clipping short. (2)")
                        break
                    non_silent_wave += non_silent_seg

            aseg = non_silent_wave

            # 3. if no proper silence found for clipping
            if len(aseg) > 12000:
                aseg = aseg[:12000]
                show_info("Audio is over 15s, clipping short. (3)")

        aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
        aseg.export(f.name, format="wav")
        ref_audio = f.name

    # Compute a hash of the reference audio file
    with open(ref_audio, "rb") as audio_file:
        audio_data = audio_file.read()
        audio_hash = hashlib.md5(audio_data).hexdigest()

    if not ref_text.strip():
        if audio_hash in ref_audio_cache:
            # Use cached asr transcription
            show_info("Using cached reference text...")
            ref_text = ref_audio_cache[audio_hash]
        else:
            show_info("No reference text provided, transcribing reference audio...")
            ref_text = transcribe_fn(ref_audio)
            # Cache the transcribed text (not caching custom ref_text, enabling users to do manual tweak)
            ref_audio_cache[audio_hash] = ref_text
    else:
        show_info("Using custom reference text...")

    # Ensure ref_text ends with a proper sentence-ending punctuation
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    print("\nref_text  ", ref_text)

    return ref_audio, ref_text


# -----------------------------------------
# Inference functions
# -----------------------------------------

def infer_process(
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    model_obj: CFM,
    vocoder: torch.nn.Module,
    mel_spec_type: str = mel_spec_type,
    show_info: Callable = print,
    progress=tqdm,
    target_rms: float = target_rms,
    cross_fade_duration: float = cross_fade_duration,
    nfe_step: int = nfe_step,
    cfg_strength: float = cfg_strength,
    sway_sampling_coef: float = sway_sampling_coef,
    speed: float = speed,
    fix_duration: Optional[float] = fix_duration,
    device: str = device,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Inference process: chunk text -> infer batches."""
    # Split the input text into batches
    audio, sr = torchaudio.load(ref_audio)
    max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (22 - audio.shape[-1] / sr))
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
    for i, gt in enumerate(gen_text_batches):
        print(f"gen_text {i}", gt)
    print("\n")

    show_info(f"Generating audio in {len(gen_text_batches)} batches...")
    return next(
        infer_batch_process(
            (audio, sr),
            ref_text,
            gen_text_batches,
            model_obj,
            vocoder,
            mel_spec_type=mel_spec_type,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=device,
        )
    )


def infer_batch_process(
    ref_audio: Tuple[torch.Tensor, int],
    ref_text: str,
    gen_text_batches: List[str],
    model_obj: CFM,
    vocoder: torch.nn.Module,
    mel_spec_type: str = "vocos",
    progress=tqdm,
    target_rms: float = 0.1,
    cross_fade_duration: float = 0.15,
    nfe_step: int = 32,
    cfg_strength: float = 2.0,
    sway_sampling_coef: float = -1,
    speed: float = 1,
    fix_duration: Optional[float] = None,
    device: Optional[str] = None,
    streaming: bool = False,
    chunk_size: int = 2048,
) -> Generator:
    """Process inference in batches."""
    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    generated_waves = []
    spectrograms = []

    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    def process_batch(gen_text):
        local_speed = speed
        if len(gen_text.encode("utf-8")) < 10:
            local_speed = 0.3

        # Prepare the text
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        ref_audio_len = audio.shape[-1] // hop_length
        if fix_duration is not None:
            duration = int(fix_duration * target_sample_rate / hop_length)
        else:
            # Calculate duration
            ref_text_len = len(ref_text.encode("utf-8"))
            gen_text_len = len(gen_text.encode("utf-8"))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)

        # inference
        with torch.inference_mode():
            generated, _ = model_obj.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )
            del _

            generated = generated.to(torch.float32)  # generated mel spectrogram
            generated = generated[:, ref_audio_len:, :]
            generated = generated.permute(0, 2, 1)
            if mel_spec_type == "vocos":
                generated_wave = vocoder.decode(generated)
            elif mel_spec_type == "bigvgan":
                generated_wave = vocoder(generated)
            if rms < target_rms:
                generated_wave = generated_wave * rms / target_rms

            # wav -> numpy
            generated_wave = generated_wave.squeeze().cpu().numpy()

            if streaming:
                for j in range(0, len(generated_wave), chunk_size):
                    yield generated_wave[j : j + chunk_size], target_sample_rate
            else:
                generated_cpu = generated[0].cpu().numpy()
                del generated
                yield generated_wave, generated_cpu

    if streaming:
        for gen_text in progress.tqdm(gen_text_batches) if progress is not None else gen_text_batches:
            for chunk in process_batch(gen_text):
                yield chunk
    else:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_batch, gen_text) for gen_text in gen_text_batches]
            for future in progress.tqdm(futures) if progress is not None else futures:
                result = future.result()
                if result:
                    generated_wave, generated_mel_spec = next(result)
                    generated_waves.append(generated_wave)
                    spectrograms.append(generated_mel_spec)

        if generated_waves:
            if cross_fade_duration <= 0:
                # Simply concatenate
                final_wave = np.concatenate(generated_waves)
            else:
                # Combine all generated waves with cross-fading
                final_wave = generated_waves[0]
                for i in range(1, len(generated_waves)):
                    prev_wave = final_wave
                    next_wave = generated_waves[i]

                    # Calculate cross-fade samples, ensuring it does not exceed wave lengths
                    cross_fade_samples = int(cross_fade_duration * target_sample_rate)
                    cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

                    if cross_fade_samples <= 0:
                        # No overlap possible, concatenate
                        final_wave = np.concatenate([prev_wave, next_wave])
                        continue

                    # Overlapping parts
                    prev_overlap = prev_wave[-cross_fade_samples:]
                    next_overlap = next_wave[:cross_fade_samples]

                    # Fade out and fade in
                    fade_out = np.linspace(1, 0, cross_fade_samples)
                    fade_in = np.linspace(0, 1, cross_fade_samples)

                    # Cross-faded overlap
                    cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

                    # Combine
                    new_wave = np.concatenate(
                        [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
                    )

                    final_wave = new_wave

            # Create a combined spectrogram
            combined_spectrogram = np.concatenate(spectrograms, axis=1)

            yield final_wave, target_sample_rate, combined_spectrogram

        else:
            yield None, target_sample_rate, None


def remove_silence_for_generated_wav(filename: str) -> None:
    """Remove silence from generated wav file."""
    aseg = AudioSegment.from_file(filename)
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    aseg = non_silent_wave
    aseg.export(filename, format="wav")


def save_spectrogram(spectrogram: np.ndarray, path: str) -> None:
    """Save spectrogram as an image."""
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    plt.savefig(path)
    plt.close()
