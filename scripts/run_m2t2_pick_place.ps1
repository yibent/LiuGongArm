param(
    [ValidateSet('so101','franka')][string]$Robot='so101',
    [ValidateSet('asset','profile')][string]$ContactMaterial='asset',
    [string]$Label='red cube',
    [string]$Destination='blue square',
    [ValidateSet('m2t2','graspgenx')][string]$GraspBackend='m2t2',
    [ValidateSet('region','tray','socket','rack')][string]$Fixture='region',
    [ValidateSet('on','inside','insert','hang')][string]$Relation='on',
    [ValidateRange(0,2)][int]$Clutter=0,
    [ValidateRange(0.20,0.30)][double]$FixtureX=0.25,
    [ValidateRange(-0.24,-0.15)][double]$FixtureY=-0.198,
    [string]$Output='',
    [switch]$RecordVideo,
    [switch]$NoHeadless,
    [switch]$FineOnly,
    [string]$CaseJson='',
    [int]$M2T2Port=5580,
    [ValidateRange(0,.02)][double]$M2T2SurfaceRangeM=.02,
    [ValidateRange(.15,.5)][double]$M2T2SceneRadiusM=.2,
    [int]$LocatorPort=5570
)
$ErrorActionPreference='Stop'
$env:MR_LIU_ROBOT=$Robot
$m2Root=Split-Path -Parent $PSScriptRoot
$m2Python=Join-Path $m2Root '_envs\m2t2\Scripts\python.exe'
$m2Vision=Join-Path $m2Root '_envs\vision\Scripts\python.exe'
if(-not $Output){$Output=Join-Path $m2Root ('output\m2t2\'+(Get-Date -Format 'yyyyMMdd_HHmmss'))}
if(-not [IO.Path]::IsPathRooted($Output)){$Output=Join-Path $m2Root $Output}
if((Test-Path -LiteralPath $Output) -and (Get-ChildItem -LiteralPath $Output -Force | Select-Object -First 1)){
    throw 'Use a new output directory; previous evidence is never overwritten'
}
New-Item -ItemType Directory -Path $Output -Force | Out-Null
$m2Owned=@()
$m2Exit=2
$m2Started=Get-Date
$m2PreviousRecordDir=$env:BUSAGENT_M2T2_RECORD_DIR
$env:BUSAGENT_M2T2_RECORD_DIR=Join-Path $Output 'model_requests'
try {
    foreach($service in @(
        @{port=$M2T2Port;python=$m2Python;script='serve_m2t2.py';name='m2t2'},
        @{port=$LocatorPort;python=$m2Vision;script='serve_semantic_locator.py';name='locator'})) {
        $health=$null
        try{$health=Invoke-RestMethod "http://127.0.0.1:$($service.port)/health" -TimeoutSec 2}catch{}
        if($null -eq $health){
            $process=Start-Process -FilePath $service.python -ArgumentList @('-u',(Join-Path $PSScriptRoot $service.script),'--port',"$($service.port)") `
                -WorkingDirectory $m2Root -WindowStyle Hidden -PassThru `
                -RedirectStandardOutput (Join-Path $Output ($service.name+'_stdout.log')) `
                -RedirectStandardError (Join-Path $Output ($service.name+'_stderr.log'))
            $m2Owned+=$process
            for($i=0;$i -lt 240;$i++){
                if($process.HasExited){throw "$($service.name) exited during startup"}
                try{$health=Invoke-RestMethod "http://127.0.0.1:$($service.port)/health" -TimeoutSec 1;break}catch{Start-Sleep -Milliseconds 250}
            }
        }
        if($null -eq $health){throw "$($service.name) startup timed out"}
        if($service.name -eq 'm2t2' -and ($health.backend -ne 'm2t2' -or $health.protocol -ne 1)){throw 'Invalid M2T2 service'}
        if($service.name -eq 'locator' -and ($health.service -ne 'busagent-semantic-locator' -or $health.protocol -ne 1)){throw 'Invalid locator service'}
    }
    $sceneView = 'oblique'
    $m2Args=@((Join-Path $PSScriptRoot 'run_fine_grasp_demo.py'),'--backend','m2t2','--place-backend','m2t2',
        '--m2t2-url',"http://127.0.0.1:$M2T2Port",'--m2t2-surface-range-m',"$M2T2SurfaceRangeM",
        '--m2t2-scene-radius-m',"$M2T2SceneRadiusM",'--place-label',$Destination,'--place-fixture',$Fixture,
        '--place-relation',$Relation,'--locator-port',"$LocatorPort",'--scene-view',$sceneView,
        '--place-fixture-x',"$FixtureX",'--place-fixture-y',"$FixtureY",'--place-clutter',"$Clutter",
        '--recovery','active','--contact-material',$ContactMaterial,'--output',$Output)
    if(-not $FineOnly){$m2Args+=@('--label',$Label,'--localization-mode','florence')}
    if($RecordVideo){$m2Args+='--record-video'}
    if($NoHeadless){$m2Args+='--no-headless'}
    if($CaseJson){$m2Args+=@('--case-json',$CaseJson)}
    $env:OMNI_KIT_ACCEPT_EULA='YES'
    if($GraspBackend -eq 'graspgenx'){
        # Explicit hybrid ablation, never an automatic fallback presented as M2T2.
        $baselineArgs=@('-Robot',$Robot,'-ContactMaterial',$ContactMaterial,'-Backend','graspgenx','-PlaceBackend','m2t2','-M2T2Url',"http://127.0.0.1:$M2T2Port",
            '-M2T2SurfaceRangeM',"$M2T2SurfaceRangeM",
            '-M2T2SceneRadiusM',"$M2T2SceneRadiusM",
            '-PlaceLabel',$Destination,'-PlaceFixture',$Fixture,'-PlaceRelation',$Relation,
            '-PlaceFixtureX',"$FixtureX",'-PlaceFixtureY',"$FixtureY",'-PlaceClutter',"$Clutter",
            '-LocatorPort',"$LocatorPort",'-SceneView',$sceneView,'-Recovery','active','-Output',$Output)
        if(-not $FineOnly){$baselineArgs+=@('-Label',$Label,'-LocalizationMode','florence')}
        if($RecordVideo){$baselineArgs+='-RecordVideo'}
        if($NoHeadless){$baselineArgs+='-NoHeadless'}
        if($CaseJson){$baselineArgs+=@('-CaseJson',$CaseJson)}
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'run_fine_grasp_demo.ps1') @baselineArgs
    } else {
        & 'D:\isaac\env_isaacsim60\python.exe' @m2Args
    }
    $m2Exit=$LASTEXITCODE
    $report=Join-Path $Output 'place_report.json'
    if(-not (Test-Path -LiteralPath $report)){$m2Exit=2}
    else {
        $result=(Get-Content -LiteralPath $report -Raw | ConvertFrom-Json).result
        if(-not ($result.success -and $result.released -and $result.verified)){$m2Exit=2}
    }
    $graspReport=Join-Path $Output 'report.json'
    if(-not (Test-Path -LiteralPath $graspReport)){$m2Exit=2}
    elseif(-not (Get-Content -LiteralPath $graspReport -Raw | ConvertFrom-Json).result.success){$m2Exit=2}
} finally {
    if($null -eq $m2PreviousRecordDir){Remove-Item Env:BUSAGENT_M2T2_RECORD_DIR -ErrorAction SilentlyContinue}
    else{$env:BUSAGENT_M2T2_RECORD_DIR=$m2PreviousRecordDir}
    foreach($process in $m2Owned){
        Get-CimInstance Win32_Process | Where-Object {
            $_.ParentProcessId -eq $process.Id -and ($_.CommandLine -like '*serve_m2t2.py*' -or $_.CommandLine -like '*serve_semantic_locator.py*')
        } | ForEach-Object {Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue}
        if(-not $process.HasExited){Stop-Process -Id $process.Id -ErrorAction SilentlyContinue}
    }
    $summary=[ordered]@{robot=$Robot;contact_material=$ContactMaterial;grasp_backend=$GraspBackend;place_backend='m2t2';complete_success=($m2Exit -eq 0);
        process_exit_code=$m2Exit;elapsed_s=((Get-Date)-$m2Started).TotalSeconds;
        input_surface_range_m=$M2T2SurfaceRangeM;place_scene_radius_m=$M2T2SceneRadiusM;
        grasp=$null;place=$null;physical_environment='Isaac Sim';real_robot_test=$false}
    foreach($entry in @(@{file='report.json';key='grasp'},@{file='place_report.json';key='place'})){
        $path=Join-Path $Output $entry.file
        if(Test-Path -LiteralPath $path){
            $item=(Get-Content -LiteralPath $path -Raw | ConvertFrom-Json).result
            $summary[$entry.key]=@{success=$item.success;failure=$item.failure;phase=$item.phase;
                released=$item.released;verified=$item.verified}
        }
    }
    $summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Output 'run_summary.json') -Encoding UTF8
}
exit $m2Exit
