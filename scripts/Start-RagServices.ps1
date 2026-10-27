$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$runDirectory = Join-Path $projectRoot ".run"
$stateFile = Join-Path $runDirectory "services.json"

if (-not (Test-Path $python)) {
    throw "Python environment not found at $python. Create the project .venv and install requirements.txt first."
}
if (Test-Path $stateFile) {
    throw "A launcher state file already exists at $stateFile. Stop the recorded services or inspect it before starting again."
}
if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue) {
    throw "Port 8000 is already in use. Stop the existing backend before running this launcher."
}

New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
$backendLog = Join-Path $runDirectory "backend.log"
$backendErrorLog = Join-Path $runDirectory "backend-error.log"
$backendProcess = $null

try {
    $backendProcess = Start-Process `
        -FilePath $python `
        -ArgumentList @("-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000") `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $backendLog `
        -RedirectStandardError $backendErrorLog `
        -WindowStyle Hidden `
        -PassThru

    $deadline = (Get-Date).AddSeconds(90)
    do {
        if ($backendProcess.HasExited) {
            throw "FastAPI exited during startup. Inspect .run\backend-error.log."
        }
        try {
            $null = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -TimeoutSec 3
            break
        } catch {
            Start-Sleep -Seconds 2
        }
    } while ((Get-Date) -lt $deadline)

    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -TimeoutSec 3
    } catch {
        throw "FastAPI did not become healthy within 90 seconds. Inspect .run\backend-error.log."
    }

    $state = [ordered]@{
        backendPid = $backendProcess.Id
    }
    $state | ConvertTo-Json | Set-Content -Path $stateFile -Encoding UTF8
    Write-Output "FastAPI is healthy at http://127.0.0.1:8000."
    Write-Output "Start ngrok separately with: ngrok http 8000"
    Write-Output "Stop these services with: .\scripts\Stop-RagServices.ps1"
} catch {
    if ($backendProcess -and -not $backendProcess.HasExited) {
        Stop-Process -Id $backendProcess.Id -ErrorAction SilentlyContinue
    }
    throw
}
