# INCIDENT 2026-05-12 — launch path inconsistency (4 failed attempts)

## Status

ROOT CAUSE IDENTIFIED — RESOLUTION DEFERRED TO PHASE 1.5
(`scripts/launch_run9_vm.sh` unified patch). Workaround for Run 9
launch was a manual bootstrap step (`mv` + `rm -rf` + `ln -s`) executed
ad-hoc on each retry; documented here for the canonical fix.

## Summary

Four sequential launch attempts on Vast.ai instance 36588175 failed
during preflight or early training because the Run 9 toolchain has
inconsistent path conventions across its components:

- `scripts/preflight_vm.sh` and `scripts/preflight_check.py` use
  **repo-relative** paths (e.g., `data/processed/clinvar_grch38.parquet`)
- `scripts/launch_run9_vm.sh` and `scripts/run9_ablations.py` use
  **absolute** paths under `/workspace/{data,outputs}/` (e.g.,
  `SPLITS_DIR=/workspace/outputs/run9_ready/splits`)
- Vast.ai SCP destination is the cloned repo at
  `/workspace/genomic-variant-classifier/`, NOT `/workspace/`

These three conventions cannot all be true simultaneously without
bootstrap-time symlinks. The launch script assumed they were already
true; preflight reported missing files; the operator added symlinks
between retries; the symlinks failed in subtle ways (see Cause column
below).

## Cross-references

- `docs/sessions/SESSION_2026-05-12.md` §Failed bullet "Launch attempt
  2/3/4" and §Learned rules 1, 3, 7 (the 7 standing rules earned this
  session).
- `docs/CHANGELOG.md` 2026-05-12 entry — section §Failed and §Fixed.
- `docs/incidents/INCIDENT_2026-05-12_vastai-destroy-interactive.md`
  and `INCIDENT_2026-05-12_cnn1d-pickle-nested-class.md` — same launch
  session, different failure modes.
- Memory entry: *"Vast.ai launch lessons 2026-05-12: (a) SCP
  destinations inside cloned repo; if /workspace/{data,outputs}/ in
  launch script — bootstrap `mv` + `rm -rf` + `ln -s` symlinks. (b)
  `ln -s` does NOT replace existing real dirs; `rm -rf` first."*

## Timeline of failed attempts

| Attempt | Time (UTC) | Failure mode | Root cause |
|--|--|--|--|
| 1 | 2026-05-12 ~15:30 | `preflight_vm.sh` reported missing ClinVar VCF + torch_geometric ImportError | Workflow-aware preflight not yet committed (commit `8a3785a` landed mid-attempt) |
| 2 | ~16:15 | Training script aborted with "FileNotFoundError: /workspace/outputs/run9_ready/splits/X_train.parquet" | Data was SCP'd to `/workspace/genomic-variant-classifier/outputs/run9_ready/splits/`, launch script expected `/workspace/outputs/run9_ready/splits/` |
| 3 | ~17:00 | Training script wrote to wrong output dir; SCP-back retrieved empty `outputs/` | Operator manually copied data into `/workspace/` to satisfy attempt-2 absolute paths; this conflicted with the launch script's `cd /workspace/genomic-variant-classifier` + assumption that `data/` and `outputs/` symlinks already existed |
| 4 | ~18:30 | `ln -s /workspace/genomic-variant-classifier/data /workspace/data` placed the symlink INSIDE an existing `/workspace/data/` directory created by attempt 3 (rather than replacing it). Training read from one path, wrote to another, both pointing at different content. | Operator did `ln -s` without first `rm -rf` of the target. The `ln` command silently creates `/workspace/data/data` when `/workspace/data/` already exists as a real directory. |
| Launch (final) | 2026-05-12 ~20:13 | Successful launch | Operator executed the corrected bootstrap: `rm -rf /workspace/data /workspace/outputs && ln -s /workspace/genomic-variant-classifier/data /workspace/data && ln -s /workspace/genomic-variant-classifier/outputs /workspace/outputs` |

## Evidence

### Path mismatch in source

`scripts/launch_run9_vm.sh`:

```bash
cd /workspace/genomic-variant-classifier   # repo root
# ...
SPLITS_DIR=/workspace/outputs/run9_ready/splits   # ABSOLUTE
OUT_BASE=/workspace/outputs/run9                  # ABSOLUTE
```

`scripts/preflight_vm.sh` (per session doc reference):

```bash
test -f data/processed/clinvar_grch38.parquet  # REPO-RELATIVE
```

`scripts/run9_ablations.py` invocation (from `launch_run9_vm.sh`):

```bash
python scripts/run9_ablations.py \
    --splits-dir "$SPLITS_DIR" \    # ABSOLUTE
    ...
    --output-dir "$OUT_BASE/$ABL" \  # ABSOLUTE
```

The launch script CDs into the repo but then passes absolute paths
that exist outside the repo. The preflight uses repo-relative paths
that are valid only inside the repo's CWD. The two cannot both be
correct without a symlink bridge.

### The `ln -s` silent-trap (attempt 4)

```bash
# Operator executed:
ln -s /workspace/genomic-variant-classifier/data /workspace/data

# Result if /workspace/data already exists as a directory:
# /workspace/data/data -> /workspace/genomic-variant-classifier/data
# (the symlink is created INSIDE the existing directory, not as a replacement)
```

This is documented Unix `ln -s` behaviour: if the second argument is
an existing directory, the symlink is created inside it with the
basename of the first argument. No error is emitted.

## Root cause

**Inconsistent path conventions across the launch toolchain.** The
preflight scripts and the launch script were authored at different
times under different assumptions. No single document specifies "the
launch convention is X" and no test validates that all components
agree.

The compounding factor is that Vast.ai's PyTorch image auto-clones
the repo to `/workspace/<repo-name>/`, but the launch script was
written as if `/workspace/` itself were the working directory.

## Remediation

### Phase 1.5 unified patch for `scripts/launch_run9_vm.sh`

Adopt **single convention: all paths repo-relative**, with the
working directory anchored to the repo root via `cd "$REPO_ROOT"` at
the top of the script:

```bash
# At top of launch_run9_vm.sh
REPO_ROOT="${REPO_ROOT:-/workspace/genomic-variant-classifier}"
cd "$REPO_ROOT" || { echo "FATAL: repo not at $REPO_ROOT"; exit 1; }

# Then all path vars become repo-relative
SPLITS_DIR="outputs/run9_ready/splits"
OUT_BASE="outputs/run9"
```

Drop the `/workspace/{data,outputs}/` bootstrap altogether. The
preflight scripts already work repo-relative; the launch script
should match.

If for some reason absolute paths under `/workspace/` are still
required (e.g., container scratch mounted at `/workspace/`),
the launch script must explicitly create the symlinks as part of its
bootstrap, with `rm -rf` of any pre-existing target:

```bash
# Defensive symlink bootstrap
for d in data outputs; do
    if [ ! -L "/workspace/$d" ]; then
        rm -rf "/workspace/$d"
        ln -s "$REPO_ROOT/$d" "/workspace/$d"
    fi
done
```

### Operational workaround (until patch lands)

For any Vast.ai launch before the patch ships, the operator must
manually execute the bootstrap immediately after SCP completion and
before invoking `launch_run9_vm.sh`:

```bash
cd /workspace/genomic-variant-classifier
rm -rf /workspace/data /workspace/outputs
ln -s /workspace/genomic-variant-classifier/data /workspace/data
ln -s /workspace/genomic-variant-classifier/outputs /workspace/outputs
ls -la /workspace/data /workspace/outputs   # must show -> symlinks
```

## Lessons

- **Single path convention per toolchain.** A launch toolchain composed
  of N scripts must agree on one path convention. Repo-relative is the
  natural choice when all scripts assume `cd "$REPO_ROOT"` first.
- **Bootstrap steps are part of the toolchain.** If a setup step is
  required between SCP and launch, it must live IN the launch script,
  not in the operator's head or a separate runbook page. Memory of
  bootstrap steps does not survive a 4-attempt-in-2-hours launch loop.
- **`ln -s` is hazardous in retry loops.** Always `rm -rf` the
  destination before symlinking, especially in scripts that may run
  more than once.
- **Each failed launch cost ~15 min of debug time and ~$0.12 of GPU
  billing.** Four failed attempts on a $0.473/hr instance = ~$0.50
  burned before training started. Acceptable on a per-incident basis;
  the standing rule is to prevent the same class of bug in future
  launches.

## Sign-off

INCIDENT moves to RESOLVED when:
- `scripts/launch_run9_vm.sh` is patched to unified repo-relative
  paths (or defensive bootstrap symlinks)
- A clean Vast.ai launch from cold-start succeeds without manual
  intervention between SCP and `bash scripts/launch_run9_vm.sh`
- The patched script is exercised at least once in a dry-run
  (`SKIP_TRAINING=1`) and once in a real launch.
