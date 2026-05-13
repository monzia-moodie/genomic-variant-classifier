---
date: 2026-05-13
severity: medium (temporary data loss, 2-hour delay; no permanent damage)
status: resolved (commit ac64665)
related_session: docs/sessions/SESSION_2026-05-13.md
related_rules: memory rule 28 (sub-items 7, 8, 9)
related_commits: ac64665
---

# INCIDENT 2026-05-13 -- Phase 1.7 apply session: PowerShell `return` no-op + missing present_files visibility

## Summary

The first Phase 1.7 apply attempt failed completely. Two failure modes
compounded: (a) Monzia did not see the `present_files` UI card for
`run10_phase1_7.zip` on initial read, so the zip was never downloaded; (b) the
PowerShell apply block used `return` after `Write-Error` to guard the missing
zip, but `return` at prompt scope is a no-op for control flow, so the block
ran to completion and the unconditional `Write-Host "[OK] ..."` lines printed
false success messages.

A side-effect of the failed apply was unguarded deletion of
`logs/training/run9_master.log.partial` (the only on-disk record of the Run 9
master log at that point) BEFORE the replacement `run9_master.log.recovery.md`
existed, causing temporary data loss.

Recovery: revert stray CHANGELOG newline, rebuild `.partial` stub from chat
transcript, re-run `present_files` for the zip, second apply succeeded cleanly
(commit `ac64665`).

## Timeline

| Time (UTC, approx) | Event |
|--------------------|-------|
| T+0   | Monzia copy-pastes apply block, no zip downloaded |
| T+1   | `Get-FileHash` against non-existent zip emits `Resolve-Path` error |
| T+1   | `$actual` is empty; `$actual -ne $expected` evaluates true |
| T+1   | `Write-Error "ABORT: SHA mismatch"` fires; `return` executes (no-op) |
| T+1   | Unconditional `Write-Host "[OK] Bundle hash verified."` prints (FALSE) |
| T+2   | `Expand-Archive` fails (no zip) -- error visible |
| T+2   | `python apply_phase1_7.py` fails (no applier) -- error visible |
| T+3   | `Get-Content ... | Out-String | Add-Content` appends empty newline to CHANGELOG.md |
| T+3   | `Remove-Item run9_master.log.partial` succeeds -- data lost |
| T+4   | `git update-index --chmod=+x` fails (no file) |
| T+4   | `git commit` runs on empty stage; `git push` reports "Everything up-to-date" |
| T+5   | Monzia notices nothing landed; reports to Claude |
| T+10  | Claude diagnoses: stray newline + missing zip + .partial deleted + `return` no-op |
| T+15  | Recovery: `git checkout -- docs/CHANGELOG.md`, rebuild .partial stub |
| T+20  | Claude builds inline contingency applier (`apply_phase1_7_inline.py`); calls present_files on both zip AND inline applier |
| T+30  | Monzia finds the zip card on second look, downloads, hash matches |
| T+35  | Apply succeeds clean (one cosmetic bash -n FAIL ignored) |
| T+40  | Post-checks all green, tests pass, commit `ac64665` pushed |

## Root causes

### RC1 -- PowerShell `return` at prompt scope is a no-op

```powershell
if (-not (Test-Path $zipPath)) {
    Write-Error "ABORT: ..."
    return                        # <-- NO-OP at prompt scope
}
$actual = (Get-FileHash $zipPath).Hash   # <-- still runs; errors
Write-Host "[OK] ..." -ForegroundColor Green   # <-- still prints, FALSE
```

`return` only terminates a function or script block. Inside an `if` block at
the interactive prompt, `return` is functionally equivalent to `;` -- it does
not abort subsequent independent statements pasted in the same buffer.

Correct patterns:

```powershell
# Pattern A: throw (raises a terminating error that halts the pipeline)
if (-not (Test-Path $zipPath)) {
    throw "ABORT: $zipPath not present"
}

# Pattern B: wrap in a function
function Apply-Phase1.7 {
    if (-not (Test-Path $zipPath)) {
        Write-Error "ABORT: ..."
        return  # <-- works inside function
    }
    # ... rest of apply ...
}
Apply-Phase1.7
```

### RC2 -- Missing `present_files` UI card visibility

The first apply session's instructions assumed the user could see the
`present_files` card. On Monzia's first read of that message, the card was not
visible (UI placement / scroll / rendering). The PowerShell that followed
referenced a path in `$env:USERPROFILE\Downloads\` that the user had no way to
populate without seeing the card.

Recovery required a second `present_files` call to expose the card a second
time. Monzia confirms: "I didn't [download] because I did not see it when I
first checked. So, I went back and downloaded the zip. Everything processed
smoothly."

### RC3 -- Unguarded delete of `.partial` fallback

```powershell
Remove-Item logs\training\run9_master.log.partial   # <-- ran before replacement existed
```

The instruction sequence assumed the prior steps (which actually failed) had
created `run9_master.log.recovery.md`. They had not. The delete ran anyway
because PowerShell did not abort the block on prior errors.

Correct pattern:

```powershell
if (Test-Path logs\training\run9_master.log.recovery.md) {
    Remove-Item logs\training\run9_master.log.partial -ErrorAction SilentlyContinue
}
```

## Fix applied

- Stray CHANGELOG newline reverted via `git checkout -- docs/CHANGELOG.md`.
- `.partial` stub re-created from in-chat SSH capture (988 bytes vs original
  990 -- cosmetic em-dash to `--` diff only).
- `present_files` re-called on zip; Monzia downloaded successfully.
- Apply ran clean, all post-checks green (modulo cosmetic bash -n Windows
  path-mangling).
- Commit `ac64665` pushed to `origin/main`.

## Prevention

Three sub-items appended to memory rule 28 (apply-batch hygiene):

- **28(7)** `return` at PS prompt is no-op; use `throw` or function-wrap.
- **28(8)** Never delete fallback until replacement verified on disk.
- **28(9)** `present_files` artifact: user must visually confirm UI card
  before running disk-assuming commands.

## Related artifacts

- `apply_phase1_7_inline.py` (54020 bytes, SHA-256 `1d2e66b9...59aeae`) --
  the inline contingency applier built during recovery, never used. Kept as
  reference for the Rule 29 pattern: for any multi-artifact apply, prefer
  single-file inline Python with base64-embedded artifacts and per-blob SHA
  verification over zip + script.
- `run10_phase1_7.zip` (SHA-256 `B2AFEED5...EDD8047`) -- the bundle that
  eventually landed via attempt 2.

## Verification

- HEAD `ac64665` is descended from `e07e3d8` (Phase 1.5e baseline).
- `git log --oneline -1` shows the Phase 1.7 commit.
- `pytest tests/unit/ -q` returns 501 passed.
- `Select-String -Path scripts\preflight_vm.sh -Pattern "^# 9\.|^# 10\.|^# 11\.|^# 12\.|^# Summary"`
  returns 5 matches in line order 225 / 244 / 261 / 276 / 310.
- All three Phase 1.7 bash/md files have 0 CR bytes on disk.
