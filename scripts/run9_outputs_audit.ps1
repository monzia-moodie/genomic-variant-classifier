# scripts/run9_outputs_audit.ps1
# ==============================
# Exhaustively scans for Run 9 training artifacts and AUROC values across
# all plausible locations on the local filesystem and the Drive mirror.
#
# Usage (from C:\Projects\genomic-variant-classifier):
#   .\scripts\run9_outputs_audit.ps1 > outputs\run9_audit_2026-05-13.txt 2>&1
#
# Then send the output file. The audit covers:
#   1. Run 9 output directories on local disk
#   2. Greppable files (logs, JSON, CSV, markdown) for AUROC mentions
#   3. Drive mirror via rclone
#   4. Recent nohup/tmux capture logs
#   5. PowerShell command history for Run 9 launch session
#   6. Untracked files git might be hiding

$ErrorActionPreference = "Continue"
$repoRoot = "C:\Projects\genomic-variant-classifier"
Set-Location $repoRoot

function Write-Section($title) {
    Write-Host ""
    Write-Host "=== $title ===" -ForegroundColor Cyan
}

Write-Section "Repo state"
git rev-parse HEAD
git rev-parse --abbrev-ref HEAD
git status --short

Write-Section "Run 9 output directories (existence + tree)"
$candidates = @(
    "outputs\run9",
    "outputs\run9_ready",
    "outputs\run9_dryrun",
    "outputs\run10_dryrun",
    "models\v1",
    "models\run9",
    "logs",
    "logs\training",
    "agent_data",
    "runs",
    "runs\smoke"
)
foreach ($d in $candidates) {
    if (Test-Path $d) {
        Write-Host "[EXISTS] $d" -ForegroundColor Green
        try {
            $files = Get-ChildItem -LiteralPath $d -Recurse -File -ErrorAction SilentlyContinue
            if ($files) {
                $files | Select-Object @{N="Path";E={$_.FullName.Replace($repoRoot + "\","")}},
                                       @{N="MB";E={[math]::Round($_.Length/1MB,3)}},
                                       @{N="Modified";E={$_.LastWriteTime.ToString("yyyy-MM-dd HH:mm")}} |
                    Sort-Object Path | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
            } else {
                Write-Host "  (empty)" -ForegroundColor DarkGray
            }
        } catch {
            Write-Host "  ERROR listing: $_" -ForegroundColor Red
        }
    } else {
        Write-Host "[MISSING] $d" -ForegroundColor Yellow
    }
}

Write-Section "Files modified between 2026-05-11 and 2026-05-13 (potential Run 9 artifacts)"
$since = (Get-Date "2026-05-11").Date
$until = (Get-Date "2026-05-13").AddDays(1).Date
Get-ChildItem -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object {
        $_.LastWriteTime -ge $since -and $_.LastWriteTime -le $until -and
        $_.FullName -notmatch "\\\.git\\|\\\.venv|\\__pycache__\\|\\node_modules\\|\\\.vscode"
    } |
    Select-Object @{N="Path";E={$_.FullName.Replace($repoRoot + "\","")}},
                  @{N="MB";E={[math]::Round($_.Length/1MB,3)}},
                  @{N="Modified";E={$_.LastWriteTime.ToString("yyyy-MM-dd HH:mm")}} |
    Sort-Object Modified | Format-Table -AutoSize | Out-String -Width 200 | Write-Host

Write-Section "Grep for OOF AUROC and 0.99* numbers across small text files"
$patterns = @(
    "OOF.*AUROC",
    "blend.*AUROC",
    "Nelder",
    "blend_weights",
    "ENSEMBLE_STACKER",
    "0\.991[0-9]"
)
Get-ChildItem -Recurse -File -Include "*.log","*.txt","*.json","*.md","*.csv" -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Length -lt 10MB -and
        $_.FullName -notmatch "\\\.git\\|\\\.venv|\\__pycache__\\|\\node_modules\\"
    } |
    ForEach-Object {
        $f = $_
        foreach ($p in $patterns) {
            try {
                $matches = Select-String -LiteralPath $f.FullName -Pattern $p -ErrorAction SilentlyContinue
                foreach ($m in $matches) {
                    [pscustomobject]@{
                        File = $f.FullName.Replace($repoRoot + "\","")
                        LineNumber = $m.LineNumber
                        Pattern = $p
                        Line = $m.Line.Trim().Substring(0, [Math]::Min(140, $m.Line.Trim().Length))
                    }
                }
            } catch {}
        }
    } | Sort-Object File, LineNumber | Format-Table -AutoSize -Wrap | Out-String -Width 220 | Write-Host

Write-Section "Drive mirror via rclone (agent_data/ namespace)"
try {
    rclone lsl "genvarcla:agent_data/" 2>&1 | Select-Object -First 60 | ForEach-Object { Write-Host $_ }
} catch {
    Write-Host "rclone not available or remote 'genvarcla' not configured: $_" -ForegroundColor Yellow
}

Write-Section "Drive mirror — top-level genvarcla:"
try {
    rclone lsd "genvarcla:" 2>&1 | Select-Object -First 30 | ForEach-Object { Write-Host $_ }
} catch {
    Write-Host "rclone lsd failed: $_" -ForegroundColor Yellow
}

Write-Section "Drive mirror — search for 'run9' filenames"
try {
    rclone lsf -R "genvarcla:" --include "*run9*" 2>&1 | Select-Object -First 60 | ForEach-Object { Write-Host $_ }
} catch {
    Write-Host "rclone search failed: $_" -ForegroundColor Yellow
}

Write-Section "PowerShell command history (lines mentioning run9 or ensemble or OOF)"
try {
    $hist = (Get-PSReadLineOption).HistorySavePath
    if (Test-Path $hist) {
        Write-Host "History file: $hist"
        Get-Content $hist | Select-String -Pattern "run9|ensemble|OOF|AUROC|0\.991" |
            Select-Object -Last 30 | ForEach-Object { Write-Host $_ }
    } else {
        Write-Host "No PS history file at $hist" -ForegroundColor Yellow
    }
} catch {
    Write-Host "PS history scan failed: $_" -ForegroundColor Yellow
}

Write-Section "Downloads folder (in case Run 9 outputs were saved there)"
$dl = "$env:USERPROFILE\Downloads"
if (Test-Path $dl) {
    Get-ChildItem -LiteralPath $dl -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match "run9|ensemble|metrics|oof|auroc" -and $_.LastWriteTime -gt (Get-Date "2026-05-11") } |
        Select-Object @{N="Path";E={$_.FullName}},
                      @{N="MB";E={[math]::Round($_.Length/1MB,3)}},
                      @{N="Modified";E={$_.LastWriteTime.ToString("yyyy-MM-dd HH:mm")}} |
        Format-Table -AutoSize | Out-String -Width 200 | Write-Host
}

Write-Section "Untracked git files (might be Run 9 outputs not yet added)"
git ls-files --others --exclude-standard | Where-Object { $_ -match "run9|outputs|metrics|json" } | ForEach-Object { Write-Host $_ }

Write-Section "Audit complete"
Write-Host "If OOF AUROCs were displayed only in chat (not on disk), they may be recoverable from:"
Write-Host "  1. Vast.ai console scrollback (instance 36588175 — destroyed, may be gone)"
Write-Host "  2. Local PowerShell scrollback from the launch session (2026-05-12)"
Write-Host "  3. ssh client buffer if a terminal multiplexer was used (tmux/screen capture-pane history)"
Write-Host "  4. SESSION_2026-05-12.md has the headline numbers; the per-model table is the gap."
