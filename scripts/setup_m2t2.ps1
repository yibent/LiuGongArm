param(
    [string]$Python = "python",
    [switch]$SkipInstall,
    [switch]$SkipWeights
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VendorRoot = Join-Path $ProjectRoot "_vendor\M2T2"
$ModelRoot = Join-Path $ProjectRoot "_models\m2t2"
$ExpectedHashes = @{
    "m2t2.pth" = "E35C3CB11E06F46C5D406BDFC756BC06F48B256DD6D638408D9A5FF13DEB97FB"
    "m2t2_language.pth" = "8CC96DAF6A50CDC3E52BB27BB163DCF78CDEB64C0972D9E7D51B24B70E1CF5FA"
}
New-Item -ItemType Directory -Force -Path $ModelRoot | Out-Null

if (-not (Test-Path (Join-Path $VendorRoot "m2t2\m2t2.py"))) {
    New-Item -ItemType Directory -Force -Path (Split-Path $VendorRoot) | Out-Null
    git clone --depth 1 https://github.com/NVlabs/M2T2.git $VendorRoot
}

if (-not $SkipInstall) {
    & $Python -c "import torch; assert torch.cuda.is_available(), 'M2T2 requires a CUDA-enabled PyTorch environment'"
    if ($LASTEXITCODE -ne 0) { throw 'CUDA-enabled PyTorch was not found; install a matching CUDA torch before pointnet2_ops' }
    & $Python -m pip install -r (Join-Path $VendorRoot "requirements.txt")
    & $Python -m pip install (Join-Path $VendorRoot "pointnet2_ops")
    & $Python -m pip install -e $VendorRoot
}

if (-not $SkipWeights) {
    $checkpoint = Join-Path $ModelRoot "m2t2.pth"
    if (-not (Test-Path $checkpoint)) {
        Invoke-WebRequest -Uri "https://huggingface.co/wentao-yuan/m2t2/resolve/main/m2t2.pth" -OutFile $checkpoint
    }
    $language = Join-Path $ModelRoot "m2t2_language.pth"
    if (-not (Test-Path $language)) {
        Invoke-WebRequest -Uri "https://huggingface.co/wentao-yuan/m2t2/resolve/main/m2t2_language.pth" -OutFile $language
    }
}

foreach ($name in $ExpectedHashes.Keys) {
    $path = Join-Path $ModelRoot $name
    if (-not (Test-Path -LiteralPath $path)) {
        throw "M2T2 weight is missing: $path"
    }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($actual -ne $ExpectedHashes[$name]) {
        throw "M2T2 weight hash mismatch for $name (expected $($ExpectedHashes[$name]), got $actual)"
    }
}

Write-Host "M2T2 checkout: $VendorRoot"
Write-Host "M2T2 weights:  $ModelRoot"
Write-Host "M2T2 weight SHA-256: verified"
Write-Host "Start server with: $Python scripts/serve_m2t2.py --checkpoint $ModelRoot/m2t2.pth"
