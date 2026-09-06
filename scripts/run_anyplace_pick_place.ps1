param(
    [ValidateSet('so101','franka')][string]$Robot='franka',
    [ValidateSet('asset','profile')][string]$ContactMaterial='profile',
    [string]$Label='red cube',
    [string]$CaseJson='',
    [switch]$InitCurrentOrientation,
    [string]$Destination='blue square',
    [ValidateSet('region','tray','socket','rack')][string]$Fixture='region',
    [ValidateSet('on','inside','insert','hang')][string]$Relation='on',
    [ValidateRange(0,2)][int]$Clutter=0,
    [ValidateRange(0.20,0.30)][double]$FixtureX=0.25,
    [ValidateRange(-0.24,-0.15)][double]$FixtureY=-0.198,
    [string]$Output='',
    [switch]$RecordVideo,
    [switch]$NoHeadless
)
$ErrorActionPreference='Stop'
if (-not $Output) {$Output='output/anyplace/'+(Get-Date -Format 'yyyyMMdd_HHmmss')}
$anyArgs=@('-Robot',$Robot,'-ContactMaterial',$ContactMaterial,'-GraspLabel',$Label,'-Destination',$Destination,'-Fixture',$Fixture,'-Relation',$Relation,
           '-PlaceBackend','anyplace','-FixtureX',"$FixtureX",'-FixtureY',"$FixtureY",'-Clutter',"$Clutter",'-Output',$Output)
if ($CaseJson) {$anyArgs+=@('-CaseJson',$CaseJson)}
if ($InitCurrentOrientation) {$anyArgs+='-AnyPlaceInitCurrentOrientation'}
if ($RecordVideo) {$anyArgs+='-RecordVideo'}
if ($NoHeadless) {$anyArgs+='-NoHeadless'}
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'run_fine_place.ps1') @anyArgs
exit $LASTEXITCODE
