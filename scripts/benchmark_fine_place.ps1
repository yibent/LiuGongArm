param(
    [string]$Manifest = 'config/fine_place_comparison.json',
    [string]$Output = '',
    [switch]$RecordVideo
)
$ErrorActionPreference='Stop'
$benchRoot=Split-Path -Parent $PSScriptRoot
if (-not [IO.Path]::IsPathRooted($Manifest)) {$Manifest=Join-Path $benchRoot $Manifest}
if (-not $Output) {$Output=Join-Path $benchRoot ('output/fine_place/comparison_'+(Get-Date -Format 'yyyyMMdd_HHmmss'))}
if (-not [IO.Path]::IsPathRooted($Output)) {$Output=Join-Path $benchRoot $Output}
if ((Test-Path -LiteralPath $Output) -and (Get-ChildItem -LiteralPath $Output -Force | Select-Object -First 1)) {
    throw 'Use a new/empty comparison directory; evidence is never overwritten'
}
$matrix=Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
if (-not $matrix.cases) {throw 'Comparison manifest has no cases'}
$names=@{}
foreach ($case in $matrix.cases) {
    if ($case.name -notmatch '^[a-z0-9_]+$' -or $names.ContainsKey($case.name)) {throw 'Invalid or duplicate case name'}
    $names[$case.name]=$true
}
New-Item -ItemType Directory -Path $Output -Force | Out-Null
Copy-Item -LiteralPath $Manifest -Destination (Join-Path $Output 'manifest.json')
$benchCommit=(& git -C $benchRoot rev-parse HEAD)
$benchDirty=(& git -C $benchRoot status --porcelain)
$results=@()
foreach ($case in $matrix.cases) {
    $run=Join-Path $Output $case.name
    $started=Get-Date
    Write-Host "[FinePlace benchmark] $($case.name)"
    $runArgs=@('-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $PSScriptRoot 'run_fine_place.ps1'),
        '-ReleaseMaxAgeS',"$($case.release_max_age_s)",'-Fixture',$case.fixture,
        '-Destination',$case.destination,'-Relation',$case.relation,
        '-FixtureX',"$($case.x)",'-FixtureY',"$($case.y)",'-Output',$run)
    if ($RecordVideo) {$runArgs+='-RecordVideo'}
    & powershell.exe @runArgs
    $runExit=$LASTEXITCODE
    $place=$null;$grasp=$null
    if (Test-Path -LiteralPath (Join-Path $run 'place_report.json')) {
        $place=Get-Content -LiteralPath (Join-Path $run 'place_report.json') -Raw | ConvertFrom-Json
    }
    if (Test-Path -LiteralPath (Join-Path $run 'report.json')) {
        $grasp=Get-Content -LiteralPath (Join-Path $run 'report.json') -Raw | ConvertFrom-Json
    }
    $results+=[PSCustomObject]@{
        name=$case.name;exit_code=$runExit;wall_s=((Get-Date)-$started).TotalSeconds
        grasp_success=($null -ne $grasp -and $grasp.result.success)
        success=($null -ne $place -and $place.result.success -and $place.result.released -and $place.result.verified)
        released=($null -ne $place -and $place.result.released)
        failure=$(if ($place) {$place.result.failure} elseif ($grasp) {"grasp:$($grasp.result.failure)"} else {'missing_report'})
        message=$place.result.message;metrics=$place.result.metrics;configuration=$case
    }
    $summary=[PSCustomObject]@{git_commit=$benchCommit;initial_git_status=$benchDirty;results=$results}
    [IO.File]::WriteAllText((Join-Path $Output 'summary.json'),($summary | ConvertTo-Json -Depth 20))
}
$results | Select-Object name,success,released,failure,wall_s | Format-Table -AutoSize
if (@($results | Where-Object {-not $_.success}).Count) {exit 2}
