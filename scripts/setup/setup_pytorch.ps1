<#
.SYNOPSIS
    Automatically detect GPU and install appropriate PyTorch version.

.DESCRIPTION
    This script detects your NVIDIA GPU and installs the correct PyTorch version
    with CUDA support. It handles RTX 50/40/30/20 series and GTX GPUs.

.PARAMETER DryRun
    Show what would be installed without actually installing.

.PARAMETER CudaVersion
    Force specific CUDA version (e.g., "12.8", "12.4", "12.1", "11.8", "cpu").

.PARAMETER Force
    Force reinstall even if PyTorch is already installed.

.EXAMPLE
    .\setup_pytorch.ps1
    .\setup_pytorch.ps1 -DryRun
    .\setup_pytorch.ps1 -CudaVersion "12.4"
    .\setup_pytorch.ps1 -Force
#>

param(
    [switch]$DryRun,
    [string]$CudaVersion = "",
    [switch]$Force
)

# Set UTF-8 encoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"

# Colors for output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Write-Header {
    param([string]$Title)
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
}

function Write-Success { param([string]$Message) Write-Host "[OK] $Message" -ForegroundColor Green }
function Write-Warning { param([string]$Message) Write-Host "[WARNING] $Message" -ForegroundColor Yellow }
function Write-Error { param([string]$Message) Write-Host "[ERROR] $Message" -ForegroundColor Red }
function Write-Info { param([string]$Message) Write-Host "[INFO] $Message" -ForegroundColor Cyan }

# GPU series to CUDA version mapping
$GpuCudaMapping = @{
    "RTX 50" = @{ Cuda = "12.8"; Nightly = $true; Arch = "Blackwell" }
    "RTX 40" = @{ Cuda = "12.4"; Nightly = $false; Arch = "Ada Lovelace" }
    "RTX 30" = @{ Cuda = "12.1"; Nightly = $false; Arch = "Ampere" }
    "RTX A"  = @{ Cuda = "12.1"; Nightly = $false; Arch = "Ampere" }
    "RTX 20" = @{ Cuda = "11.8"; Nightly = $false; Arch = "Turing" }
    "GTX 16" = @{ Cuda = "11.8"; Nightly = $false; Arch = "Turing" }
    "GTX 10" = @{ Cuda = "11.8"; Nightly = $false; Arch = "Pascal" }
}

function Get-GpuInfo {
    <#
    .SYNOPSIS
        Get GPU information from nvidia-smi.
    #>

    $result = @{
        Available = $false
        GpuName = ""
        DriverVersion = ""
        CudaVersion = ""
        GpuSeries = ""
        MemoryTotal = ""
    }

    # Check if nvidia-smi exists
    $nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        return $result
    }

    try {
        # Get GPU name
        $gpuName = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        if ($LASTEXITCODE -ne 0) { return $result }

        # Get driver and CUDA version from nvidia-smi output
        $nvidiaSmiOutput = & nvidia-smi 2>$null
        $cudaMatch = [regex]::Match($nvidiaSmiOutput, "CUDA Version:\s*(\d+\.\d+)")
        $driverMatch = [regex]::Match($nvidiaSmiOutput, "Driver Version:\s*(\d+\.\d+)")

        # Get memory
        $memoryInfo = & nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>$null

        $result.Available = $true
        $result.GpuName = $gpuName.Trim()
        $result.CudaVersion = if ($cudaMatch.Success) { $cudaMatch.Groups[1].Value } else { "" }
        $result.DriverVersion = if ($driverMatch.Success) { $driverMatch.Groups[1].Value } else { "" }
        $result.MemoryTotal = if ($memoryInfo) { $memoryInfo.Trim() } else { "" }

        # Detect GPU series
        foreach ($series in $GpuCudaMapping.Keys) {
            if ($gpuName -match $series) {
                $result.GpuSeries = $series
                break
            }
        }

        # Default to unknown if no match
        if (-not $result.GpuSeries) {
            $result.GpuSeries = "Unknown"
        }
    }
    catch {
        # nvidia-smi failed
        return $result
    }

    return $result
}

function Get-RecommendedCuda {
    param([string]$GpuSeries)

    if ($GpuCudaMapping.ContainsKey($GpuSeries)) {
        return $GpuCudaMapping[$GpuSeries]
    }

    # Safe default for unknown GPUs
    return @{ Cuda = "12.1"; Nightly = $false; Arch = "Unknown" }
}

function Get-PyTorchInstallCommand {
    param(
        [string]$CudaVer,
        [bool]$IsNightly
    )

    $packages = "torch torchvision torchaudio"

    switch ($CudaVer) {
        "12.8" {
            return "pip install --pre $packages --index-url https://download.pytorch.org/whl/nightly/cu128"
        }
        "12.4" {
            return "pip install $packages --index-url https://download.pytorch.org/whl/cu124"
        }
        "12.1" {
            return "pip install $packages --index-url https://download.pytorch.org/whl/cu121"
        }
        "11.8" {
            return "pip install $packages --index-url https://download.pytorch.org/whl/cu118"
        }
        "cpu" {
            return "pip install $packages"
        }
        default {
            return "pip install $packages --index-url https://download.pytorch.org/whl/cu121"
        }
    }
}

function Test-PyTorchInstalled {
    <#
    .SYNOPSIS
        Check if PyTorch is already installed and its CUDA status.
    #>

    $result = @{
        Installed = $false
        Version = ""
        CudaAvailable = $false
        CudaVersion = ""
    }

    try {
        $pythonOutput = & python -c "import torch; print(f'{torch.__version__}|{torch.cuda.is_available()}|{torch.version.cuda if torch.cuda.is_available() else \"\"}')" 2>$null
        if ($LASTEXITCODE -eq 0 -and $pythonOutput) {
            $parts = $pythonOutput.Split("|")
            $result.Installed = $true
            $result.Version = $parts[0]
            $result.CudaAvailable = $parts[1] -eq "True"
            $result.CudaVersion = $parts[2]
        }
    }
    catch {
        # PyTorch not installed
    }

    return $result
}

function Uninstall-PyTorch {
    Write-Info "Uninstalling existing PyTorch..."
    & pip uninstall torch torchvision torchaudio -y 2>$null
}

function Install-PyTorch {
    param(
        [string]$Command,
        [bool]$DryRunMode
    )

    if ($DryRunMode) {
        Write-Info "Would run: $Command"
        return $true
    }

    Write-Info "Running: $Command"
    Invoke-Expression $Command

    return $LASTEXITCODE -eq 0
}

# Main execution
Write-Header "PyTorch Setup for F5-TTS"

# Step 1: Check Python environment
Write-Host ""
Write-Info "Checking Python environment..."

$pythonCmd = Get-Command "python" -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Error "Python not found. Please install Python 3.9-3.12 first."
    exit 1
}

$pythonVersion = & python --version 2>&1
Write-Success "Python found: $pythonVersion"

# Step 2: Check GPU
Write-Host ""
Write-Info "Detecting GPU..."

$gpuInfo = Get-GpuInfo

if (-not $gpuInfo.Available) {
    Write-Warning "No NVIDIA GPU detected or nvidia-smi not found."
    Write-Host ""

    if ($CudaVersion -eq "" -or $CudaVersion -eq "cpu") {
        $CudaVersion = "cpu"
        Write-Info "Will install CPU-only PyTorch."
    }
    else {
        Write-Info "Forcing CUDA $CudaVersion as requested."
    }
}
else {
    Write-Success "GPU detected: $($gpuInfo.GpuName)"
    Write-Host "  Driver Version: $($gpuInfo.DriverVersion)" -ForegroundColor Gray
    Write-Host "  CUDA Version: $($gpuInfo.CudaVersion)" -ForegroundColor Gray
    Write-Host "  Memory: $($gpuInfo.MemoryTotal)" -ForegroundColor Gray
    Write-Host "  GPU Series: $($gpuInfo.GpuSeries)" -ForegroundColor Gray
}

# Step 3: Determine CUDA version
Write-Host ""

if ($CudaVersion -eq "") {
    # Auto-detect
    $recommended = Get-RecommendedCuda -GpuSeries $gpuInfo.GpuSeries
    $CudaVersion = $recommended.Cuda
    $isNightly = $recommended.Nightly

    Write-Info "Recommended CUDA version for $($gpuInfo.GpuSeries) ($($recommended.Arch)): $CudaVersion"

    if ($isNightly) {
        Write-Warning "RTX 50 series requires PyTorch nightly build (may be less stable)."
    }
}
else {
    Write-Info "Using specified CUDA version: $CudaVersion"
    $isNightly = $CudaVersion -eq "12.8"
}

# Step 4: Check existing PyTorch
Write-Host ""
Write-Info "Checking existing PyTorch installation..."

$pytorchStatus = Test-PyTorchInstalled

if ($pytorchStatus.Installed) {
    Write-Host "  PyTorch Version: $($pytorchStatus.Version)" -ForegroundColor Gray
    Write-Host "  CUDA Available: $($pytorchStatus.CudaAvailable)" -ForegroundColor Gray
    if ($pytorchStatus.CudaVersion) {
        Write-Host "  CUDA Version: $($pytorchStatus.CudaVersion)" -ForegroundColor Gray
    }

    if (-not $Force) {
        if ($pytorchStatus.CudaAvailable) {
            Write-Success "PyTorch with CUDA is already installed."
            Write-Host ""
            Write-Host "Use -Force to reinstall." -ForegroundColor Gray
            exit 0
        }
        elseif ($CudaVersion -ne "cpu" -and $gpuInfo.Available) {
            Write-Warning "CPU-only PyTorch detected on a GPU machine!"
            Write-Host ""
        }
    }
}
else {
    Write-Info "PyTorch is not installed."
}

# Step 5: Get install command
$installCommand = Get-PyTorchInstallCommand -CudaVer $CudaVersion -IsNightly $isNightly

# Step 6: Show summary
Write-Header "Installation Summary"
Write-Host ""
Write-Host "  GPU: $(if ($gpuInfo.Available) { $gpuInfo.GpuName } else { 'None detected' })" -ForegroundColor White
Write-Host "  CUDA Version: $CudaVersion$(if ($isNightly) { ' (nightly)' } else { '' })" -ForegroundColor White
Write-Host "  Command: $installCommand" -ForegroundColor Yellow
Write-Host ""

if ($DryRun) {
    Write-Warning "DRY RUN - No changes will be made."
    Write-Host ""
    Write-Host "To install, run without -DryRun:" -ForegroundColor Gray
    Write-Host "  .\setup_pytorch.ps1" -ForegroundColor Gray
    exit 0
}

# Step 7: Uninstall existing PyTorch if needed
if ($pytorchStatus.Installed) {
    Write-Host ""
    Uninstall-PyTorch
}

# Step 8: Install PyTorch
Write-Host ""
Write-Info "Installing PyTorch with CUDA $CudaVersion..."
Write-Host ""

$success = Install-PyTorch -Command $installCommand -DryRunMode $false

if ($success) {
    Write-Host ""
    Write-Success "PyTorch installed successfully!"
    Write-Host ""

    # Verify installation
    Write-Info "Verifying installation..."
    $newStatus = Test-PyTorchInstalled

    if ($newStatus.Installed) {
        Write-Host "  PyTorch Version: $($newStatus.Version)" -ForegroundColor Green
        Write-Host "  CUDA Available: $($newStatus.CudaAvailable)" -ForegroundColor $(if ($newStatus.CudaAvailable) { "Green" } else { "Yellow" })
        if ($newStatus.CudaVersion) {
            Write-Host "  CUDA Version: $($newStatus.CudaVersion)" -ForegroundColor Green
        }
    }

    Write-Host ""
    Write-Header "Next Steps"
    Write-Host ""
    Write-Host "  1. Install F5-TTS:" -ForegroundColor White
    Write-Host "     pip install -e ." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  2. Verify GPU detection:" -ForegroundColor White
    Write-Host "     python -m f5_tts.tools.check_device --test" -ForegroundColor Yellow
    Write-Host ""
}
else {
    Write-Host ""
    Write-Error "PyTorch installation failed!"
    Write-Host ""
    Write-Host "Try installing manually:" -ForegroundColor Gray
    Write-Host "  $installCommand" -ForegroundColor Yellow
    exit 1
}
