# F5-TTS Scripts

Windows scripts for F5-TTS setup, training, inference, and evaluation.

## Directory Structure

```
scripts/
├── setup/          # Environment setup
│   ├── setup_pytorch.ps1
│   └── setup_pytorch.bat
├── infer/          # Inference scripts
│   ├── infer_vi.ps1
│   ├── infer_vi.bat
│   ├── infer.ps1
│   └── infer.bat
├── train/          # Training scripts
│   ├── fine_tuning.ps1
│   └── fine_tuning.bat
└── tools/          # Data preparation tools
    ├── convert_sr.bat
    ├── prepare_metadata.bat
    ├── check_vocab.bat
    └── eval_batch.bat
```

## Setup

### Install PyTorch with CUDA

Before using F5-TTS, install PyTorch with the correct CUDA version:

```powershell
# Auto-detect GPU and install PyTorch
.\scripts\setup\setup_pytorch.ps1

# Preview what will be installed (dry run)
.\scripts\setup\setup_pytorch.ps1 -DryRun

# Force specific CUDA version
.\scripts\setup\setup_pytorch.ps1 -CudaVersion "12.4"

# Force reinstall
.\scripts\setup\setup_pytorch.ps1 -Force
```

### GPU -> CUDA Mapping

| GPU Series | CUDA Version |
|------------|--------------|
| RTX 50xx (5070, 5080, 5090) | 12.8 (nightly) |
| RTX 40xx (4060, 4070, 4080, 4090) | 12.4 |
| RTX 30xx (3060, 3070, 3080, 3090) | 12.1 |
| RTX 20xx / GTX 16xx | 11.8 |

## Inference

### Vietnamese (Recommended)

```powershell
# Using config file (avoids UTF-8 issues)
.\scripts\infer\infer_vi.ps1

# Custom config
.\scripts\infer\infer_vi.ps1 -Config "configs/my_config.toml"
```

Edit `configs/infer_vi.toml` to change text:

```toml
ref_text = "text tham chiếu"
gen_text = "text cần sinh giọng nói"
```

### General Inference

```powershell
# With parameters
.\scripts\infer\infer.ps1 -GenText "Hello world" -Speed 0.9

# Full options
.\scripts\infer\infer.ps1 `
    -RefAudio "ref/vi_ref_1.wav" `
    -RefText "reference text" `
    -GenText "text to generate" `
    -Model "F5TTS_Base" `
    -Speed 1.0
```

## Training

### Fine-tuning Pipeline

```powershell
# Run full pipeline (stages 0-5)
.\scripts\train\fine_tuning.ps1 -Stage 0 -StopStage 5

# Run only training (stage 5)
.\scripts\train\fine_tuning.ps1 -Stage 5 -StopStage 5

# Custom dataset
.\scripts\train\fine_tuning.ps1 -DatasetName "my_dataset" -ExpName "F5TTS_Base"
```

### Pipeline Stages

| Stage | Description |
|-------|-------------|
| 0 | Convert audio sample rate to 24kHz |
| 1 | Prepare metadata (metadata.csv, vocab.txt) |
| 2 | Check vocabulary against pretrained model |
| 3 | Extend model embeddings for new vocabulary |
| 4 | Extract features (prepare_csv_wavs) |
| 5 | Run fine-tuning |

## Tools

### Data Preparation

```powershell
# Convert audio sample rate
.\scripts\tools\convert_sr.bat

# Prepare metadata
.\scripts\tools\prepare_metadata.bat

# Check vocabulary
.\scripts\tools\check_vocab.bat

# Batch evaluation
.\scripts\tools\eval_batch.bat
```

## Verification

Before running, verify GPU is detected:

```powershell
# Check PyTorch CUDA status
python -m f5_tts.tools.check_pytorch

# Check device info
python -m f5_tts.tools.check_device --test
```

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
.\scripts\infer\infer_vi.ps1

# Use second GPU
$env:CUDA_VISIBLE_DEVICES = "1"
```

### UTF-8 Encoding

For Vietnamese text, always use config files (`infer_vi.ps1`) instead of passing text as command-line arguments.

### Multi-GPU Training

Edit `scripts/train/fine_tuning.ps1` and uncomment the `accelerate launch` section.
