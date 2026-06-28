<#
.SYNOPSIS
    GPU-aware bootstrap for LocalAIExecution (Blackwell / RTX 5090, cu128).

.DESCRIPTION
    Creates the virtual environment, installs the CUDA 12.8 (cu128) PyTorch
    build that actually drives the Blackwell sm_120 GPU, installs the package,
    verifies CUDA on the GPU (hard gate), then optionally warms the default
    model and runs a real generation smoke test.

    The hardest platform risk is retired here: standard PyPI / cu121 wheels do
    NOT support sm_120 and silently fall back to CPU. We install from the cu128
    index and, if verification fails, retry from the cu128 nightly index.

.PARAMETER Nightly
    Install torch from the cu128 *nightly* index from the start.

.PARAMETER SkipSmoke
    Skip the model warm + real generation smoke (still runs the CUDA doctor gate).

.PARAMETER Model
    Model id to warm for the smoke test (default: schnell). Gated models (dev)
    are never auto-downloaded.

.EXAMPLE
    ./scripts/bootstrap.ps1
.EXAMPLE
    ./scripts/bootstrap.ps1 -Nightly -SkipSmoke
#>
[CmdletBinding()]
param(
    [switch]$Nightly,
    [switch]$SkipSmoke,
    [string]$Model = "schnell"
)

$ErrorActionPreference = "Stop"

$StableIndex  = "https://download.pytorch.org/whl/cu128"
$NightlyIndex = "https://download.pytorch.org/whl/nightly/cu128"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Localai    = Join-Path $RepoRoot ".venv\Scripts\localai.exe"

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "BOOTSTRAP FAILED: $msg" -ForegroundColor Red; exit 1 }

# 1. Python 3.12 -------------------------------------------------------------
Write-Step "Checking Python 3.12"
$pyExe = $null
foreach ($cand in @("py -3.12", "python")) {
    try {
        $ver = & ([scriptblock]::Create("$cand --version")) 2>&1
        if ($ver -match "3\.12") { $pyExe = $cand; break }
    } catch { }
}
if (-not $pyExe) { Fail "Python 3.12 not found on PATH (required)." }
Write-Host "Using: $pyExe ($ver)"

# 2. Virtual environment -----------------------------------------------------
Write-Step "Creating virtual environment (.venv)"
if (-not (Test-Path $VenvPython)) {
    & ([scriptblock]::Create("$pyExe -m venv .venv"))
    if ($LASTEXITCODE -ne 0) { Fail "venv creation failed." }
} else {
    Write-Host ".venv already exists - reusing."
}
& $VenvPython -m pip install --upgrade pip --quiet

# 3. GPU detection -----------------------------------------------------------
Write-Step "Detecting NVIDIA GPU"
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) { Fail "nvidia-smi not found - no NVIDIA GPU/driver detected." }
$gpu = & nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
Write-Host "GPU: $gpu"

# 4. Install cu128 torch -----------------------------------------------------
$index = if ($Nightly) { $NightlyIndex } else { $StableIndex }
Write-Step "Installing PyTorch from cu128 index: $index"
if ($Nightly) {
    & $VenvPython -m pip install --pre torch --index-url $index
} else {
    & $VenvPython -m pip install torch --index-url $index
}
if ($LASTEXITCODE -ne 0) { Fail "torch install failed from $index." }

# 5. Install the package (pulls diffusers/transformers/etc.) ------------------
Write-Step "Installing the localai package (editable)"
& $VenvPython -m pip install -e .
if ($LASTEXITCODE -ne 0) { Fail "package install failed." }

# 6. Verify CUDA on the GPU (hard gate) + tensor smoke ------------------------
Write-Step "Verifying CUDA on the GPU (doctor)"
& $Localai doctor
if ($LASTEXITCODE -ne 0) {
    if (-not $Nightly) {
        Write-Host "Stable cu128 verification failed - retrying from the nightly index..." -ForegroundColor Yellow
        & $VenvPython -m pip install --pre torch --index-url $NightlyIndex --force-reinstall
        if ($LASTEXITCODE -ne 0) { Fail "nightly torch install failed." }
        & $Localai doctor
        if ($LASTEXITCODE -ne 0) { Fail "CUDA verification failed even with the nightly cu128 build." }
    } else {
        Fail "CUDA verification failed with the nightly cu128 build."
    }
}

# 7. Warm the default model + real generation smoke --------------------------
if ($SkipSmoke) {
    Write-Step "Skipping model warm/smoke (-SkipSmoke)"
} else {
    Write-Step "Warming '$Model' + running a real generation smoke (first run downloads weights)"
    $outDir = Join-Path $RepoRoot "outputs"
    & $Localai generate "a smoke-test photo of a single blue cube on a white background" `
        --model $Model --steps 4 --width 512 --height 512 --seed 0 --output-dir $outDir
    if ($LASTEXITCODE -ne 0) { Fail "generation smoke failed (exit $LASTEXITCODE)." }
}

Write-Step "BOOTSTRAP COMPLETE"
Write-Host "Try:  .\.venv\Scripts\localai.exe generate `"a serene mountain lake at dawn`"" -ForegroundColor Green
