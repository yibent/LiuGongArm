param(
    [string]$Output='',
    [double[]]$CubePosition=@(0.3,0.3,0.02575),
    [double[]]$TargetPosition=@(-0.3,-0.3,0.02575),
    [double[]]$CubeSize=@(0.0515,0.0515,0.0515),
    [double[]]$CubeOrientation=@(1.0,0.0,0.0,0.0),
    [double]$GripperYawDeg=0.0,
    [switch]$RecordVideo,
    [switch]$IndustrialScene,
    [int]$MoveTargetPhase=-1,
    [int]$MoveTargetStep=30,
    [double[]]$MoveTargetDelta=@(0.08,0.0,0.0),
    [switch]$NoHeadless
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
if(-not $Output){$Output=Join-Path $root ('output\franka\official_pick_place_'+(Get-Date -Format 'yyyyMMdd_HHmmss'))}
if(-not [IO.Path]::IsPathRooted($Output)){$Output=Join-Path $root $Output}
$args=@(
    (Join-Path $PSScriptRoot 'run_official_franka_pick_place_legacy.py'),
    '--output',$Output,
    '--cube-position',"$($CubePosition[0])","$($CubePosition[1])","$($CubePosition[2])",
    '--cube-orientation',"$($CubeOrientation[0])","$($CubeOrientation[1])","$($CubeOrientation[2])","$($CubeOrientation[3])",
    '--target-position',"$($TargetPosition[0])","$($TargetPosition[1])","$($TargetPosition[2])",
    '--cube-size',"$($CubeSize[0])","$($CubeSize[1])","$($CubeSize[2])",
    '--gripper-yaw-deg',"$GripperYawDeg",
    '--move-target-phase',"$MoveTargetPhase",
    '--move-target-step',"$MoveTargetStep",
    '--move-target-delta',"$($MoveTargetDelta[0])","$($MoveTargetDelta[1])","$($MoveTargetDelta[2])"
)
if($NoHeadless){$args+='--no-headless'}
if($RecordVideo){$args+='--record-video'}
if($IndustrialScene){$args+='--industrial-scene'}
$env:OMNI_KIT_ACCEPT_EULA='YES'
& 'D:\isaac\env_isaacsim60\python.exe' @args
exit $LASTEXITCODE
