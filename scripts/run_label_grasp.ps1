param(
    [string]$Label = 'red cube',
    [ValidateSet('florence', 'florence_yoloe')][string]$LocalizationMode = 'florence_yoloe',
    [ValidateSet('geometric', 'graspgenx', 'm2t2')][string]$Backend = 'graspgenx',
    [ValidateSet('off', 'assisted', 'active')][string]$Recovery = 'active',
    [ValidateSet('fine_grasp', 'tabletop_wide')][string]$WristCameraProfile = 'fine_grasp',
    [string]$CaseJson = '', [string]$Output = '',
    [ValidateRange(-0.05, 0.05)][double]$TestCoarseShiftM = 0,
    [int]$LocatorPort = 5570, [switch]$CoarseOnly, [switch]$RecordVideo, [switch]$NoHeadless
)
$ErrorActionPreference = 'Stop'
$labelRoot = Split-Path -Parent $PSScriptRoot
$labelPython = Join-Path $labelRoot '_envs\vision\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $labelPython)) {
    # Keep the checked-in overlay as the preferred deployment path, while
    # allowing the repository's existing local venv to run the service during
    # development on machines that have not created _envs\vision yet.
    $labelPython = Join-Path $labelRoot '.venv\python.exe'
}
if (-not (Test-Path -LiteralPath $labelPython)) {
    throw 'Vision Python was not found; run scripts/setup_vision.ps1 or create .venv first'
}
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $labelRoot ('output\label_grasp\' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
}
if (-not [System.IO.Path]::IsPathRooted($Output)) { $Output = Join-Path $labelRoot $Output }
if ((Test-Path -LiteralPath $Output) -and (Get-ChildItem -LiteralPath $Output -Force | Select-Object -First 1)) {
    throw 'Use a new output directory to preserve previous tests'
}
New-Item -ItemType Directory -Path $Output -Force | Out-Null
$labelService = $null
try {
    try { $health = Invoke-RestMethod "http://127.0.0.1:$LocatorPort/health" -TimeoutSec 2 } catch { $health = $null }
    if ($null -eq $health) {
        $labelService = Start-Process -FilePath $labelPython -ArgumentList @(
            (Join-Path $PSScriptRoot 'serve_semantic_locator.py'), '--port', "$LocatorPort") `
            -WorkingDirectory $labelRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $Output 'locator_stdout.log') `
            -RedirectStandardError (Join-Path $Output 'locator_stderr.log')
        for ($i=0; $i -lt 60; $i++) {
            if ($labelService.HasExited) { throw 'Local semantic locator exited; inspect locator logs' }
            try { $health = Invoke-RestMethod "http://127.0.0.1:$LocatorPort/health" -TimeoutSec 1; break } catch { Start-Sleep -Milliseconds 250 }
        }
    }
    if ($health.service -ne 'busagent-semantic-locator' -or $health.protocol -ne 1) { throw 'Invalid semantic locator health response' }
    $argsForDemo = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
        (Join-Path $PSScriptRoot 'run_fine_grasp_demo.ps1'), '-Label', $Label,
        '-LocatorPort', "$LocatorPort", '-LocalizationMode', $LocalizationMode,
        '-Backend', $Backend, '-Recovery', $Recovery, '-WristCameraProfile', $WristCameraProfile,
        '-SceneView', 'oblique', '-Output', $Output)
    if ($CaseJson) { $argsForDemo += @('-CaseJson', $CaseJson) }
    if ($CoarseOnly) { $argsForDemo += '-CoarseOnly' }
    if ($TestCoarseShiftM -ne 0) { $argsForDemo += @('-TestCoarseShiftM', "$TestCoarseShiftM") }
    if ($RecordVideo) { $argsForDemo += '-RecordVideo' }
    if ($NoHeadless) { $argsForDemo += '-NoHeadless' }
    # Isaac writes non-fatal renderer warnings to stderr. PowerShell promotes
    # native stderr to a terminating NativeCommandError under Stop; capture
    # the stream while using the child process exit code as the result.
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & powershell @argsForDemo 2>&1 | Tee-Object -FilePath (Join-Path $Output 'console.log')
    $labelExit = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
} finally {
    # A venv launcher can create a second python.exe on Windows. Only reap
    # our exact child, never other servers or an already-running shared locator.
    if ($null -ne $labelService) {
        Get-CimInstance Win32_Process | Where-Object {
            $_.ParentProcessId -eq $labelService.Id -and $_.CommandLine -like '*serve_semantic_locator.py*'
        } | ForEach-Object { Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue }
        if (-not $labelService.HasExited) { Stop-Process -Id $labelService.Id -ErrorAction SilentlyContinue }
    }
}
exit $labelExit
