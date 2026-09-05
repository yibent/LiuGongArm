param(
    [string]$GraspLabel = 'red cube',
    [string]$Destination = 'blue square',
    [ValidateSet('region','tray')][string]$Fixture = 'region',
    [ValidateSet('on','inside')][string]$Relation = 'on',
    [string]$Output = '',
    [switch]$RecordVideo,
    [switch]$NoHeadless,
    [int]$LocatorPort = 5570,
    [ValidateSet('geometric','anyplace')][string]$PlaceBackend='geometric',
    [int]$AnyPlacePort=5590
)
$ErrorActionPreference='Stop'
$placeRoot=Split-Path -Parent $PSScriptRoot
$placePython=Join-Path $placeRoot '_envs\vision\Scripts\python.exe'
if (-not $Output) { $Output=Join-Path $placeRoot ('output\fine_place\'+(Get-Date -Format 'yyyyMMdd_HHmmss')) }
if (-not [IO.Path]::IsPathRooted($Output)) { $Output=Join-Path $placeRoot $Output }
if ((Test-Path -LiteralPath $Output) -and (Get-ChildItem -LiteralPath $Output -Force | Select-Object -First 1)) {
    throw 'Use a new output directory; previous evidence is never overwritten'
}
New-Item -ItemType Directory -Path $Output -Force | Out-Null
$placeService=$null
$placeDemo=$null
$placeModelIds=@()
$placeExit=1
$anyplaceService=$null
try {
    if ($PlaceBackend -eq 'anyplace') {
        try {$anyHealth=Invoke-RestMethod "http://127.0.0.1:$AnyPlacePort/health" -TimeoutSec 2} catch {$anyHealth=$null}
        if ($null -eq $anyHealth) {
            $anyplaceService=Start-Process -FilePath (Join-Path $placeRoot '_envs\anyplace\Scripts\python.exe') -ArgumentList @(
                '-u',(Join-Path $PSScriptRoot 'serve_anyplace.py'),'--port',"$AnyPlacePort",'--record-dir',(Join-Path $Output 'model_requests')) `
                -WorkingDirectory $placeRoot -WindowStyle Hidden -PassThru `
                -RedirectStandardOutput (Join-Path $Output 'anyplace_stdout.log') `
                -RedirectStandardError (Join-Path $Output 'anyplace_stderr.log')
            for ($i=0;$i -lt 120;$i++) {
                if ($anyplaceService.HasExited) {throw 'AnyPlace service failed; inspect logs'}
                try {$anyHealth=Invoke-RestMethod "http://127.0.0.1:$AnyPlacePort/health" -TimeoutSec 1;break} catch {Start-Sleep -Milliseconds 250}
            }
        }
        if ($anyHealth.service -ne 'busagent-anyplace' -or $anyHealth.protocol -ne 1) {throw 'Invalid AnyPlace service'}
    }
    try { $health=Invoke-RestMethod "http://127.0.0.1:$LocatorPort/health" -TimeoutSec 2 } catch { $health=$null }
    if ($null -eq $health) {
        $placeService=Start-Process -FilePath $placePython -ArgumentList @(
            (Join-Path $PSScriptRoot 'serve_semantic_locator.py'),'--port',"$LocatorPort") `
            -WorkingDirectory $placeRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $Output 'locator_stdout.log') `
            -RedirectStandardError (Join-Path $Output 'locator_stderr.log')
        for ($i=0;$i -lt 120;$i++) {
            if ($placeService.HasExited) { throw 'Florence service failed; inspect logs' }
            try {$health=Invoke-RestMethod "http://127.0.0.1:$LocatorPort/health" -TimeoutSec 1;break} catch {Start-Sleep -Milliseconds 250}
        }
    }
    if ($health.service -ne 'busagent-semantic-locator' -or $health.protocol -ne 1) { throw 'Invalid Florence service' }
    $placeArgs=@('-NoProfile','-ExecutionPolicy','Bypass','-File',
        (Join-Path $PSScriptRoot 'run_fine_grasp_demo.ps1'),'-Backend','graspgenx','-Recovery','active',
        '-Label',$GraspLabel,'-LocalizationMode','florence',
        '-SceneView','oblique','-PlaceLabel',$Destination,'-PlaceFixture',$Fixture,'-PlaceRelation',$Relation,
        '-LocatorPort',"$LocatorPort",'-Output',$Output,
        '-PlaceBackend',$PlaceBackend,'-AnyPlaceUrl',"http://127.0.0.1:$AnyPlacePort")
    if ($RecordVideo) {$placeArgs+='-RecordVideo'}
    if ($NoHeadless) {$placeArgs+='-NoHeadless'}
    # Do not hold an inherited native pipe open through orphaned model helpers
    # if Isaac terminates its launcher. Read diagnostics from the per-run logs.
    $quotedArgs=$placeArgs | ForEach-Object { '"'+($_ -replace '"','\"')+'"' }
    $placeDemo=Start-Process -FilePath 'powershell.exe' -ArgumentList $quotedArgs `
        -WorkingDirectory $placeRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $Output 'console.log') `
        -RedirectStandardError (Join-Path $Output 'console_stderr.log')
    $null=$placeDemo.Handle
    Write-Host "[BusAgent] FinePlace logs: $Output"
    while (-not $placeDemo.WaitForExit(1000)) {
        $placeModelIds+=@(Get-CimInstance Win32_Process | Where-Object {
            $_.ParentProcessId -eq $placeDemo.Id -and $_.CommandLine -like '*serve_graspgenx.py*'
        } | Select-Object -ExpandProperty ProcessId)
    }
    $placeExit=$placeDemo.ExitCode
    # PowerShell/.NET may lose a reaped child's exit status. A successful grasp
    # report alone is never a successful place command.
    $placeReportPath=Join-Path $Output 'place_report.json'
    if (-not (Test-Path -LiteralPath $placeReportPath)) { $placeExit=2 }
    else {
        $placeReport=Get-Content -LiteralPath $placeReportPath -Raw | ConvertFrom-Json
        if (-not ($placeReport.result.success -and $placeReport.result.released -and $placeReport.result.verified)) { $placeExit=2 }
        elseif ($null -eq $placeExit) { $placeExit=1 }
    }
} finally {
    if ($null -ne $anyplaceService) {
        Get-CimInstance Win32_Process | Where-Object {
            $_.ParentProcessId -eq $anyplaceService.Id -and $_.CommandLine -like '*serve_anyplace.py*'
        } | ForEach-Object {Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue}
        if (-not $anyplaceService.HasExited) {Stop-Process -Id $anyplaceService.Id -ErrorAction SilentlyContinue}
    }
    foreach ($ownedId in ($placeModelIds | Select-Object -Unique)) {
        $owned=Get-CimInstance Win32_Process -Filter "ProcessId=$ownedId" -ErrorAction SilentlyContinue
        if ($owned -and $owned.CommandLine -like '*serve_graspgenx.py*') { Stop-Process -Id $ownedId -ErrorAction SilentlyContinue }
    }
    if ($null -ne $placeService) {
        Get-CimInstance Win32_Process | Where-Object {
            $_.ParentProcessId -eq $placeService.Id -and $_.CommandLine -like '*serve_semantic_locator.py*'
        } | ForEach-Object {Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue}
        if (-not $placeService.HasExited) {Stop-Process -Id $placeService.Id -ErrorAction SilentlyContinue}
    }
}
exit $placeExit
