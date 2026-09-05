param(
    [string]$BasePython = 'D:\isaac\env_isaacsim60\python.exe',
    [switch]$SkipModels
)
$ErrorActionPreference = 'Stop'
$visionRoot = Split-Path -Parent $PSScriptRoot
$visionEnv = Join-Path $visionRoot '_envs\vision'
$visionPython = Join-Path $visionEnv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $BasePython)) { throw "Base Python missing: $BasePython" }
if (-not (Test-Path -LiteralPath $visionEnv)) {
    & $BasePython -m venv --system-site-packages $visionEnv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create vision overlay' }
}
if (-not (Test-Path -LiteralPath $visionPython)) { throw "Invalid existing vision environment: $visionEnv" }
& $visionPython -c 'import torch; assert torch.__version__ == "2.11.0+cu128", "This local installer targets torch 2.11.0+cu128; select a matching base environment first"'
if ($LASTEXITCODE -ne 0) { throw 'Unsupported base PyTorch; refusing to replace it implicitly' }
# Process-scoped caches on the project drive, not the nearly-full system drive.
$env:PIP_CACHE_DIR = Join-Path $visionRoot '.cache\pip-vision'
$env:YOLO_CONFIG_DIR = Join-Path $visionRoot '.cache\ultralytics'
New-Item -ItemType Directory -Path $env:YOLO_CONFIG_DIR -Force | Out-Null
# The base environment has CPU-only torchvision. Install CUDA ops in the
# overlay, without reinstalling torch or mutating the original environment.
& $visionPython -m pip install --no-deps 'torchvision==0.26.0+cu128' --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA torchvision installation failed' }
& $visionPython -m pip install -r (Join-Path $visionRoot 'requirements-vision.txt')
if ($LASTEXITCODE -ne 0) { throw 'Vision dependency installation failed' }
& $visionPython -c 'import torch, torchvision, cv2; from transformers import Florence2ForConditionalGeneration; from ultralytics import YOLOE; print("CUDA:", torch.cuda.is_available(), "torch:", torch.__version__, "torchvision:", torchvision.__version__, "OpenCV:", cv2.__version__); assert hasattr(cv2, "TrackerMIL_create"); device="cuda" if torch.cuda.is_available() else "cpu"; print("NMS:", torchvision.ops.nms(torch.tensor([[0.,0.,4.,4.]],device=device), torch.ones(1,device=device),0.5))'
if ($LASTEXITCODE -ne 0) { throw 'Vision import check failed' }
if (-not $SkipModels) {
    & $visionPython (Join-Path $PSScriptRoot 'download_vision_models.py')
    if ($LASTEXITCODE -ne 0) { throw 'Model download/verification failed' }
}
Write-Host "Vision ready: $visionPython"
