$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$stateFile = Join-Path $projectRoot ".run\services.json"

if (-not (Test-Path $stateFile)) {
    Write-Output "No launcher state file found; no services were stopped."
    exit 0
}

$state = Get-Content $stateFile -Raw | ConvertFrom-Json
$targets = @(
    @{ Id = $state.backendPid; Pattern = 'python(\.exe)?.*-m uvicorn backend\.main:app' }
)
foreach ($target in $targets) {
    if (-not $target.Id) {
        continue
    }
    $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $($target.Id)" -ErrorAction SilentlyContinue
    if (-not $processInfo) {
        continue
    }
    if ($processInfo.CommandLine -notmatch $target.Pattern) {
        throw "PID $($target.Id) no longer matches the recorded RAG service command; refusing to stop it."
    }
    Stop-Process -Id $target.Id
    Write-Output "Stopped process $($target.Id)."
}

Remove-Item -LiteralPath $stateFile
Write-Output "RAG services stopped."
