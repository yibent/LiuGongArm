param(
    [ValidateSet("geometric", "graspgenx")]
    [string]$Backend = "graspgenx",
    [switch]$NoHeadless,
    [switch]$KeepOpen,
    [switch]$RecordVideo,
    [string]$CaseJson = "",
    [string]$Label = "",
    [int]$LocatorPort = 5570,
    [ValidateSet("florence", "florence_yoloe")]
    [string]$LocalizationMode = "florence_yoloe",
    [switch]$CoarseOnly,
    [double]$TestCoarseShiftM = 0,
    [ValidateRange(0, 120)]
    [double]$StartDelayS = 0,
    [int]$Port = 5556,
    [string]$Output = "",
    [switch]$Benchmark,
    [ValidateSet("single", "multiview")]
    [string]$Perception = "single",
    [ValidateSet("depth", "sam2")]
    [string]$Segmenter = "depth",
    [ValidateSet("off", "assisted", "active")]
    [string]$Recovery = "off",
    [int]$DropInitialWristFrames = 0,
    [double]$TestTargetShiftM = 0,
    [ValidateSet("overhead", "oblique")]
    [string]$SceneView = "overhead",
    [string]$Manifest = "",
    [ValidateSet("development", "acceptance")]
    [string]$Split = "development",
    [string]$Seeds = "0",
    [string]$Cases = "cube,sphere,cylinder,thin,metallic_part,apple,bottle,hammer,wrench,screwdriver,key,coffee_mug"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$IsaacPython = "D:\isaac\env_isaacsim60\python.exe"
$ModelPython = Join-Path $ProjectRoot "_envs\graspgenx\python.exe"
$ModelRoot = Join-Path $ProjectRoot "_vendor\GraspGenX"
$CheckpointRoot = Join-Path $ProjectRoot "_models\graspgenx\checkpoints\release"
$CheckpointCacheRoot = Split-Path -Parent $CheckpointRoot
$GripperConfigRoot = Join-Path $ModelRoot "ext\gripper_descriptions"
$OutputRoot = Join-Path $ProjectRoot "output\graspgenx_server"
$ServerProcess = $null
$WarmupProcess = $null

if (($KeepOpen -or $StartDelayS -gt 0) -and ($Benchmark -or -not $NoHeadless)) {
    throw "-KeepOpen and -StartDelayS require -NoHeadless and a single demo (no -Benchmark)"
}

if ([string]::IsNullOrWhiteSpace($Output)) {
    $RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RunKind = if ($Benchmark) { "unseen_benchmark" } else { "fine_grasp_runs" }
    $Output = Join-Path $ProjectRoot "output\$RunKind\${RunStamp}_${Backend}"
}
elseif (-not [System.IO.Path]::IsPathRooted($Output)) {
    $Output = Join-Path $ProjectRoot $Output
}

if (-not (Test-Path -LiteralPath $IsaacPython)) {
    throw "Isaac Python was not found at $IsaacPython"
}

function Test-LocalPort([int]$TargetPort) {
    $Client = [System.Net.Sockets.TcpClient]::new()
    try {
        $Connect = $Client.BeginConnect("127.0.0.1", $TargetPort, $null, $null)
        return $Connect.AsyncWaitHandle.WaitOne(250) -and $Client.Connected
    }
    catch {
        return $false
    }
    finally {
        $Client.Dispose()
    }
}

try {
    if ($Backend -eq "graspgenx" -and -not (Test-LocalPort $Port)) {
        foreach ($RequiredPath in @($ModelPython, $ModelRoot, $CheckpointRoot, $GripperConfigRoot)) {
            if (-not (Test-Path -LiteralPath $RequiredPath)) {
                throw "GraspGenX local dependency is missing: $RequiredPath"
            }
        }
        New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
        $ServerScript = Join-Path $PSScriptRoot "serve_graspgenx.py"
        $ServerArgs = @(
            $ServerScript,
            "--config", $CheckpointRoot,
            "--assets_dir", (Join-Path $ModelRoot "assets"),
            "--host", "127.0.0.1",
            "--port", "$Port"
        )
        # GraspGenX imports bootstrap its own model and gripper repositories unless
        # both locations are explicit. The child inherits these values.
        $env:GRASPGENX_CHECKPOINT_DIR = $CheckpointCacheRoot
        $env:GRASPGENX_GRIPPER_CFG_DIR = $GripperConfigRoot
        $ServerProcess = Start-Process `
            -FilePath $ModelPython `
            -ArgumentList $ServerArgs `
            -WorkingDirectory $ModelRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $OutputRoot "stdout.log") `
            -RedirectStandardError (Join-Path $OutputRoot "stderr.log") `
            -PassThru
        for ($Attempt = 0; $Attempt -lt 240; $Attempt++) {
            if ($ServerProcess.HasExited) {
                throw "GraspGenX server exited during startup; inspect $OutputRoot"
            }
            if (Test-LocalPort $Port) {
                break
            }
            Start-Sleep -Milliseconds 250
        }
        if (-not (Test-LocalPort $Port)) {
            throw "GraspGenX server did not listen on port $Port within 60 seconds"
        }
        $WarmupProcess = Start-Process `
            -FilePath $IsaacPython `
            -ArgumentList @(
                (Join-Path $PSScriptRoot "warm_graspgenx_server.py"),
                "--port", "$Port"
            ) `
            -WorkingDirectory $ProjectRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $OutputRoot "warmup_stdout.log") `
            -RedirectStandardError (Join-Path $OutputRoot "warmup_stderr.log") `
            -PassThru
        if (-not $WarmupProcess.WaitForExit(45000)) {
            throw "GraspGenX warmup timed out; inspect $OutputRoot"
        }
        $WarmupProcess.WaitForExit()
        $Warmup = Get-Content -LiteralPath (Join-Path $OutputRoot "warmup_stdout.log") -Raw | ConvertFrom-Json
        if ($Warmup.status -ne "ready") {
            throw "GraspGenX warmup did not report ready; inspect $OutputRoot"
        }
        if ($null -ne $WarmupProcess.ExitCode -and $WarmupProcess.ExitCode -ne 0) {
            # Isaac's bundled Python can return a Windows shutdown status after
            # its client has already received and validated the full response.
            # The structured response is the readiness contract; retain the
            # exit status as a diagnostic instead of discarding a healthy model.
            Write-Warning "Warmup returned ready but exited with code $($WarmupProcess.ExitCode)"
        }
    }

    $env:OMNI_KIT_ACCEPT_EULA = "YES"
    Write-Host "[BusAgent] FineGrasp output: $Output"
    if ($Benchmark) {
        $NormalizedCases = $Cases -replace "\s+", ","
        $DemoArgs = @(
            (Join-Path $PSScriptRoot "run_fine_grasp_benchmark.py"),
            "--isaac-python", $IsaacPython,
            "--backend", $Backend,
            "--graspgenx-port", "$Port",
            "--seeds", $Seeds,
            "--cases", $NormalizedCases,
            "--output", $Output
        )
    }
    else {
        $DemoArgs = @(
            (Join-Path $PSScriptRoot "run_fine_grasp_demo.py"),
            "--backend", $Backend,
            "--graspgenx-port", "$Port",
            "--output", $Output
        )
    }
    if ($NoHeadless) {
        $DemoArgs += "--no-headless"
    }
    if ($KeepOpen) { $DemoArgs += "--keep-open" }
    if ($RecordVideo) { $DemoArgs += "--record-video" }
    if (-not [string]::IsNullOrWhiteSpace($CaseJson)) { $DemoArgs += @("--case-json", $CaseJson) }
    if (-not [string]::IsNullOrWhiteSpace($Label)) {
        if ($Benchmark) { throw "-Label is a single-scene entry, not the old fine-only benchmark" }
        $DemoArgs += @("--label", $Label, "--locator-port", "$LocatorPort", "--localization-mode", $LocalizationMode)
    }
    if ($CoarseOnly) { $DemoArgs += "--coarse-only" }
    if ($TestCoarseShiftM -ne 0) { $DemoArgs += @("--test-coarse-shift-m", "$TestCoarseShiftM") }
    if ($StartDelayS -gt 0) { $DemoArgs += @("--start-delay-s", "$StartDelayS") }
    $DemoArgs += @("--perception", $Perception)
    $DemoArgs += @("--segmenter", $Segmenter)
    $DemoArgs += @("--recovery", $Recovery, "--drop-initial-wrist-frames", "$DropInitialWristFrames",
                   "--test-target-shift-m", "$TestTargetShiftM", "--scene-view", $SceneView)
    if (-not [string]::IsNullOrWhiteSpace($Manifest)) {
        if (-not $Benchmark) { throw "-Manifest requires -Benchmark" }
        $DemoArgs += @("--manifest", $Manifest, "--split", $Split)
    }
    & $IsaacPython @DemoArgs
    exit $LASTEXITCODE
}
finally {
    if ($null -ne $WarmupProcess -and -not $WarmupProcess.HasExited) {
        Stop-Process -Id $WarmupProcess.Id
        $WarmupProcess.WaitForExit(5000)
    }
    if ($null -ne $ServerProcess -and -not $ServerProcess.HasExited) {
        Stop-Process -Id $ServerProcess.Id
        $ServerProcess.WaitForExit(5000)
    }
}
