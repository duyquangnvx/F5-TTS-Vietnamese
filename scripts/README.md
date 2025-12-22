# F5-TTS Scripts

Windows scripts for F5-TTS setup, training, inference, and evaluation.

## Scripts Overview

| Script | Description |
|--------|-------------|
| `setup_pytorch.ps1` | **Setup**: Auto-detect GPU and install PyTorch with CUDA |
| `setup_pytorch.bat` | Setup (batch version) |
| `infer_vi.ps1` | Vietnamese inference using config file (recommended) |
| `infer_vi.bat` | Vietnamese inference (batch version) |
| `infer.ps1` | General inference with parameters |
| `infer.bat` | General inference (batch version) |
| `fine_tuning.ps1` | Fine-tuning pipeline (PowerShell) |
| `fine_tuning.bat` | Fine-tuning pipeline (batch) |
| `eval_batch.bat` | Batch evaluation |

## Setup PyTorch

Before using F5-TTS, install PyTorch with the correct CUDA version for your GPU:

```powershell
# Auto-detect GPU and install PyTorch
.\scripts\setup_pytorch.ps1

# Preview what will be installed (dry run)
.\scripts\setup_pytorch.ps1 -DryRun

# Force specific CUDA version
.\scripts\setup_pytorch.ps1 -CudaVersion "12.4"

# Force reinstall
.\scripts\setup_pytorch.ps1 -Force
```

### GPU -> CUDA Mapping

| GPU Series | CUDA Version |
|------------|--------------|
| RTX 50xx (5070, 5080, 5090) | 12.8 (nightly) |
| RTX 40xx (4060, 4070, 4080, 4090) | 12.4 |
| RTX 30xx (3060, 3070, 3080, 3090) | 12.1 |
| RTX 20xx / GTX 16xx | 11.8 |

## Quick Start

### Inference (Vietnamese)

```powershell
# Using config file (recommended - avoids UTF-8 issues)
.\scripts\infer_vi.ps1

# Custom config
.\scripts\infer_vi.ps1 -Config "configs/my_config.toml"
```

Edit `configs/infer_vi.toml` to change text:

```toml
ref_text = "text tham chiếu"
gen_text = "text cần sinh giọng nói"
```

### Inference (General)

```powershell
# With parameters
.\scripts\infer.ps1 -GenText "Hello world" -Speed 0.9

# Full options
.\scripts\infer.ps1 `
    -RefAudio "ref/vi_ref_1.wav" `
    -RefText "reference text" `
    -GenText "text to generate" `
    -Model "F5TTS_Base" `
    -Speed 1.0
```

### Fine-tuning

```powershell
# Run full pipeline (stages 0-5)
.\scripts\fine_tuning.ps1 -Stage 0 -StopStage 5

# Run only training (stage 5)
.\scripts\fine_tuning.ps1 -Stage 5 -StopStage 5

# Custom dataset
.\scripts\fine_tuning.ps1 -DatasetName "my_dataset" -ExpName "F5TTS_Base"
```

## Pipeline Stages

| Stage | Description |
|-------|-------------|
| 0 | Convert audio sample rate to 24kHz |
| 1 | Prepare metadata (metadata.csv, vocab.txt) |
| 2 | Check vocabulary against pretrained model |
| 3 | Extend model embeddings for new vocabulary |
| 4 | Extract features (prepare_csv_wavs) |
| 5 | Run fine-tuning |

## Check GPU

Before running, verify GPU is detected:

```powershell
python -m f5_tts.tools.check_device --test
```

Expected output:
```
==================================================
Device Information
==================================================
  Device Type: CUDA
  Device Name: NVIDIA GeForce RTX ...
  Memory: XX.XX GB
  Dtype: torch.float16
==================================================

[OK] CUDA is working correctly!
```

## Output Directory

All outputs are saved to the `output/` directory:

```
output/
├── inference/      # Inference outputs (.wav files)
├── eval/           # Evaluation results
└── logs/           # Training logs
```

## Configuration Files

| File | Description |
|------|-------------|
| `configs/infer_vi.toml` | Vietnamese inference config |

Create custom configs by copying and editing `infer_vi.toml`.

## Notes

### PowerShell Execution Policy

If you get an execution policy error:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### GPU Selection

```powershell
# Use specific GPU
$env:CUDA_VISIBLE_DEVICES = "0"
.\scripts\infer_vi.ps1

# Use second GPU
$env:CUDA_VISIBLE_DEVICES = "1"
```

### UTF-8 Encoding

For Vietnamese text, always use config files (`infer_vi.ps1`) instead of passing text as command-line arguments to avoid encoding issues.

### Multi-GPU Training

Edit `fine_tuning.ps1` and uncomment the `accelerate launch` section.
