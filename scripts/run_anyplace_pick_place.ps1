param(
    [string]$Label='red cube',
    [string]$Destination='blue square',
    [ValidateSet('region','tray')][string]$Fixture='region',
    [ValidateSet('on','inside')][string]$Relation='on',
    [string]$Output='',
    [switch]$RecordVideo,
    [switch]$NoHeadless
)
$ErrorActionPreference='Stop'
if (-not $Output) {$Output='output/anyplace/'+(Get-Date -Format 'yyyyMMdd_HHmmss')}
$anyArgs=@('-GraspLabel',$Label,'-Destination',$Destination,'-Fixture',$Fixture,'-Relation',$Relation,
           '-PlaceBackend','anyplace','-Output',$Output)
if ($RecordVideo) {$anyArgs+='-RecordVideo'}
if ($NoHeadless) {$anyArgs+='-NoHeadless'}
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'run_fine_place.ps1') @anyArgs
exit $LASTEXITCODE
