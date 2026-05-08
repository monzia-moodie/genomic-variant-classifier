# scripts/preflight_consolidate_package_layout.ps1
# Pre-flight gate for the consolidate-package-layout migration (HYP_consolidate-package-layout.md).
# Run BEFORE C1. Each block prints OK / FAIL / WARN / INFO. Any FAIL halts: resolve before proceeding.
#
# Usage (from repo root, in .venv312):
#   pwsh -File scripts\preflight_consolidate_package_layout.ps1
# Exit codes: 0 = green (proceed), 1 = red (resolve failures).

$ErrorActionPreference = 'Continue'
$failures = @()

# 0. CWD sanity (must be at repo root)
if (-not (Test-Path .\.git) -or -not (Test-Path .\src) -or -not (Test-Path .\agent_layer)) {
    "FAIL  CWD does not look like genomic-variant-classifier repo root"
    "      Expected: .git, src, and agent_layer directories present"
    "      Actual CWD: $(Get-Location)"
    exit 1
}
"OK    CWD = $(Get-Location)"

# 1. PowerShell version (must be 7+ for consistent UTF-8 no-BOM behavior)
$psVer = $PSVersionTable.PSVersion.Major
if ($psVer -ge 7) { "OK    PowerShell $psVer" } else {
    "FAIL  PowerShell $psVer (need 7+)"; $failures += "psversion"
}

# 2. Active venv is .venv312 with Python 3.12.x
$pyVer = & python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
if ($pyVer -match '^3\.12\.') { "OK    Python $pyVer" } else {
    "FAIL  Python $pyVer (need 3.12.x)"; $failures += "pyver"
}
$venv = & python -c "import sys, os; print(os.path.basename(getattr(sys, 'prefix', '')))"
if ($venv -eq '.venv312') { "OK    venv = $venv" } else {
    "FAIL  venv = $venv (need .venv312)"; $failures += "venv"
}

# 3. Git working tree clean and on commit b74daf1 (or descendant)
$gitStatus = git status --porcelain
if (-not $gitStatus) { "OK    Working tree clean" } else {
    "FAIL  Working tree dirty:"; Write-Host $gitStatus; $failures += "gitclean"
}
$head = git rev-parse --short HEAD
"INFO  HEAD = $head (expected b74daf1 or descendant)"

# 4. src/__init__.py contains expected re-export, AND no consumers exist
if (-not (Test-Path .\src\__init__.py)) {
    "FAIL  src/__init__.py missing"; $failures += "srcinit_missing"
}
else {
    $initContent = Get-Content .\src\__init__.py -Raw
    if ($initContent -match 'from\s+src\.utils\.helpers\s+import\s+resolve_data_dir') {
        "OK    src/__init__.py contains expected re-export"
    }
    else {
        "WARN  src/__init__.py present but does not match expected content"
    }
    # Empirical consumer check
    $consumers = Get-ChildItem -Path . -Recurse -Include "*.py", "*.ipynb" `
        -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch '\\(\.git|\.venv|\.venv312|__pycache__|.*\.egg-info)\\' } |
    Select-String -Pattern '^\s*from\s+src\s+import\b|^\s*import\s+src\s*(#|$)|\bsrc\.(resolve_data_dir|__version__|__author__)\b'
    if ($consumers) {
        "FAIL  src/__init__.py re-exports have consumers:"
        $consumers | Format-Table -AutoSize
        $failures += "srcinit_consumers"
    }
    else {
        "OK    src/__init__.py re-exports have zero consumers (safe to drop)"
    }
}

# 5. setup.py present and matches expected size (587 bytes)
$setupBytes = (Get-Item .\setup.py -ErrorAction SilentlyContinue).Length
if ($null -eq $setupBytes) {
    "FAIL  setup.py missing"; $failures += "setup_missing"
}
elseif ($setupBytes -eq 587) {
    "OK    setup.py is 587 bytes"
}
else {
    "WARN  setup.py is $setupBytes bytes (spec expected 587)"
}

# 6. pyproject.toml does NOT yet exist
if (-not (Test-Path .\pyproject.toml)) { "OK    No pre-existing pyproject.toml" }
else { "FAIL  pyproject.toml already exists (spec assumes it doesn't)"; $failures += "pyproject" }

# 7. monitoring/ stub deleted (Phase 0 invariant)
if (-not (Test-Path .\monitoring)) { "OK    monitoring/ absent" }
else { "FAIL  monitoring/ still present"; $failures += "monitoring" }

# 8. egg-info inventory at repo root
$eggInfos = Get-ChildItem -Directory -Filter "*.egg-info" -ErrorAction SilentlyContinue
if ($eggInfos) {
    "INFO  egg-info dirs at root: $(($eggInfos | ForEach-Object { $_.Name }) -join ', ')"
}
else {
    "INFO  No egg-info dirs at root"
}
if ($eggInfos.Count -gt 1) {
    "WARN  Multiple egg-info dirs - pip may pick wrong one. C2 must delete all."
}

# 9. Disk space (need >= 25 GB free for C4 .bak originals)
$drive = (Get-Item .).PSDrive
$freeGB = [math]::Round($drive.Free / 1GB, 1)
if ($freeGB -ge 25) { "OK    Free space: ${freeGB} GB" }
else { "FAIL  Free space: ${freeGB} GB (need >= 25 GB)"; $failures += "disk" }

# 10. The 6 production joblibs exist at expected paths and sizes
$joblibs = @(
    @{ Path = "models\phase2_pipeline.joblib"; MB = 2029.75 },
    @{ Path = "models\phase4_pipeline.joblib"; MB = 2036.06 },
    @{ Path = "models\phase4_pipeline_calibrated.joblib"; MB = 2036.06 },
    @{ Path = "models\v1\ensemble_v1.joblib"; MB = 1344.55 },
    @{ Path = "outputs\run9_ready\models\ensemble.joblib"; MB = 1478.4 },
    @{ Path = "experiments\2026-04-04_03-39\ensemble_v1.joblib"; MB = 1344.55 }
)
foreach ($j in $joblibs) {
    if (Test-Path $j.Path) {
        $actualMB = [math]::Round((Get-Item $j.Path).Length / 1MB, 2)
        "OK    $($j.Path) - $actualMB MB (expected $($j.MB))"
    }
    else {
        "FAIL  $($j.Path) missing"; $failures += "joblib:$($j.Path)"
    }
}

# 11. Verify experiments/.../ensemble_v1.joblib SHA256 = models/v1/ensemble_v1.joblib SHA256
if ((Test-Path .\models\v1\ensemble_v1.joblib) -and (Test-Path .\experiments\2026-04-04_03-39\ensemble_v1.joblib)) {
    $h1 = (Get-FileHash -Algorithm SHA256 .\models\v1\ensemble_v1.joblib).Hash
    $h2 = (Get-FileHash -Algorithm SHA256 .\experiments\2026-04-04_03-39\ensemble_v1.joblib).Hash
    if ($h1 -eq $h2) { "OK    Both ensemble_v1.joblib copies SHA256-match: $h1" }
    else { "FAIL  ensemble_v1.joblib copies differ (h1=$h1, h2=$h2)"; $failures += "sha256_ensemble" }
}
else {
    "INFO  Skipping SHA256 compare (one or both files missing per check 10)"
}

# 12. Current bare imports work (baseline health check)
$bareOut = & python -c "import agents, config, message_bus, shared_state; print('bareOK')" 2>&1
if ($bareOut -match 'bareOK') { "OK    Bare agent_layer imports work (agents, config, message_bus, shared_state)" }
else { "FAIL  Bare imports broken: $bareOut"; $failures += "bareimports" }

# 13. src.api.main importable today
$apiOut = & python -c "import src.api.main; print('apiOK')" 2>&1
if ($apiOut -match 'apiOK') { "OK    src.api.main imports" }
else { "FAIL  src.api.main does not import: $apiOut"; $failures += "srcmain" }

# 14. Empirical inventory of `import src.X.Y` lines that the C3 sweep MUST handle (rev 2)
$importSrcHits = Get-ChildItem -Path . -Recurse -Include "*.py", "*.ipynb" `
    -ErrorAction SilentlyContinue |
Where-Object { $_.FullName -notmatch '\\(\.git|\.venv|\.venv312|__pycache__|.*\.egg-info)\\' } |
Select-String -Pattern '^\s*import\s+src\.[\w.]+'
if ($importSrcHits) {
    "INFO  $($importSrcHits.Count) lines of 'import src.X.Y' form found (must be handled by C3 patch F-5):"
    $importSrcHits | Select-Object Path, LineNumber, Line | Format-Table -AutoSize -Wrap
}
else {
    "OK    No 'import src.X.Y' lines"
}

# 15. Inventory non-.py files inside src/
$nonPy = Get-ChildItem -Path .\src -Recurse -File |
Where-Object { $_.Extension -notin '.py', '.pyc' -and $_.FullName -notmatch '__pycache__' }
"INFO  Non-.py files under src/: $($nonPy.Count)"
if ($nonPy.Count -gt 0) {
    $nonPy | Select-Object -First 20 | ForEach-Object { "      $($_.FullName)" }
    if ($nonPy.Count -gt 20) { "      ... and $($nonPy.Count - 20) more" }
}

# 16. tests/tests/ glitch
if (Test-Path .\tests\tests) {
    "WARN  tests/tests/ exists - investigate before C1"
    Get-ChildItem .\tests\tests -Recurse -File | Select-Object FullName | Format-Table -AutoSize
}
else {
    "OK    No tests/tests/ glitch"
}

# 17. Pre-commit / hook script existence
if (Test-Path .\.pre-commit-config.yaml) { "OK    .pre-commit-config.yaml exists - must be in C3 sweep" }
else { "INFO  No .pre-commit-config.yaml" }
if (Test-Path .\check_run_id_trailer.py) { "OK    check_run_id_trailer.py exists" }
elseif (Test-Path .\scripts\check_run_id_trailer.py) { "OK    scripts/check_run_id_trailer.py exists" }
else { "INFO  No check_run_id_trailer.py found" }

# 18. validate_docs.py exists
if (Test-Path .\validate_docs.py) { "OK    validate_docs.py exists - re-run after C5 doc edits" }
else { "INFO  No validate_docs.py at repo root" }

Write-Host ""
if ($failures.Count -eq 0) {
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host "PRE-FLIGHT GREEN - safe to start C1." -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor Green
    exit 0
}
else {
    Write-Host "================================================================" -ForegroundColor Red
    Write-Host "PRE-FLIGHT RED - resolve these before starting C1:" -ForegroundColor Red
    $failures | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    Write-Host "================================================================" -ForegroundColor Red
    exit 1
}