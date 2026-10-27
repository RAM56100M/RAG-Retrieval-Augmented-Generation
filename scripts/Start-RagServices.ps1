param(
    [string]$TunnelName
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$envFile = Join-Path $projectRoot ".env"
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

if ($TunnelName) {
    if (-not (Test-Path $envFile)) {
        throw "Create $envFile and configure RAG_ACCESS_PASSWORD and FRONTEND_ORIGINS before starting a public tunnel."
    }

    $envLines = Get-Content $envFile
    $passwordLine = $envLines | Where-Object { $_ -match '^\s*RAG_ACCESS_PASSWORD\s*=\s*(.+?)\s*$' } | Select-Object -First 1
    $frontendLine = $envLines | Where-Object { $_ -match '^\s*FRONTEND_ORIGINS\s*=\s*(.+?)\s*$' } | Select-Object -First 1
    $passwordValue = ""
    $frontendValue = ""
    if ($passwordLine -match '^\s*RAG_ACCESS_PASSWORD\s*=\s*(.+?)\s*$') {
        $passwordValue = $Matches[1].Trim().Trim('"').Trim("'")
    }
    if ($frontendLine -match '^\s*FRONTEND_ORIGINS\s*=\s*(.+?)\s*$') {
        $frontendValue = $Matches[1].Trim().Trim('"').Trim("'")
    }
    if ($passwordValue.Length -lt 16) {
        throw "Set a RAG_ACCESS_PASSWORD of at least 16 characters in .env before starting a tunnel."
    }
    if (-not $frontendValue.StartsWith("https://")) {
        throw "Set FRONTEND_ORIGINS to the exact HTTPS Vercel frontend origin before starting a tunnel."
    }
    $cloudflared = Get-Command cloudflared -ErrorAction SilentlyContinue
    if (-not $cloudflared) {
        throw "cloudflared was not found. Install it with: winget install --id Cloudflare.cloudflared"
    }
}

New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
$backendLog = Join-Path $runDirectory "backend.log"
$backendErrorLog = Join-Path $runDirectory "backend-error.log"
$backendProcess = $null
$tunnelProcess = $null

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

    if ($TunnelName) {
        $tunnelLog = Join-Path $runDirectory "cloudflared.log"
        $tunnelErrorLog = Join-Path $runDirectory "cloudflared-error.log"
        $tunnelProcess = Start-Process `
            -FilePath $cloudflared.Source `
            -ArgumentList @("tunnel", "run", $TunnelName) `
            -WorkingDirectory $projectRoot `
            -RedirectStandardOutput $tunnelLog `
            -RedirectStandardError $tunnelErrorLog `
            -WindowStyle Hidden `
            -PassThru
        Start-Sleep -Seconds 3
        if ($tunnelProcess.HasExited) {
            throw "Cloudflare Tunnel exited during startup. Check its name, credentials, config, and .run\cloudflared-error.log."
        }
    }

    $state = [ordered]@{
        backendPid = $backendProcess.Id
        tunnelPid = if ($tunnelProcess) { $tunnelProcess.Id } else { $null }
        tunnelName = if ($TunnelName) { $TunnelName } else { $null }
    }
    $state | ConvertTo-Json | Set-Content -Path $stateFile -Encoding UTF8
    Write-Output "FastAPI is healthy at http://127.0.0.1:8000."
    if ($TunnelName) {
        Write-Output "Cloudflare Tunnel '$TunnelName' is running. Confirm its public hostname in the Cloudflare dashboard."
    }
    Write-Output "Stop these services with: .\scripts\Stop-RagServices.ps1"
} catch {
    foreach ($process in @($tunnelProcess, $backendProcess)) {
        if ($process -and -not $process.HasExited) {
            Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
        }
    }
    throw
}
