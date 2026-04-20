# INCIDENT 2026-04-20 — Dockerfile trainer CMD passed non-existent --skip-kan flag for 11 days

## Status

**Resolved 2026-04-20 as a side-effect of commit 8f9eb60.** No rebuild
required because the trainer container was never invoked in
production. Documenting for audit-trail completeness per standing
rule #8.

## Summary

From 2026-04-09 through 2026-04-20, `Dockerfile` line 166 passed a
`--skip-kan` CLI flag to `scripts/run_phase2_eval.py` that argparse
did not accept. Any build of `genomic-variant-trainer` via
`docker build --target trainer` followed by `docker run` would have
failed immediately with `argparse: error: unrecognized arguments:
--skip-kan` and exit code 2, before any training code ran.

The condition went undetected for 11 days because Runs 6, 7, and 8
provisioned training via startup scripts on GCP, Lambda, and Vast.ai
respectively (`scripts/gcp_run6_startup.sh` and equivalents) rather
than via the Docker trainer image. The API image (Stage 2) was
unaffected — it invokes gunicorn against `src.api.main:app`, not
`run_phase2_eval.py`.

## Timeline

- **2026-04-09 06:38:02 (commit a95c9db):** `LiteratureScoutAgent`
  added with pykan watch target, ClinVar schema hash, AlphaMissense
  ETag. Relevant to this incident only because it set up the "KAN
  will return when pykan memory fix is detected" expectation that
  subsequent commits were written against.

- **2026-04-09 07:38:10 (commit 671e48d):** "fix: Dockerfile —
  venv pip in trainer, PATH in trainer stage, gnomad-constraint +
  skip-kan in trainer CMD". Added `--skip-kan` to the trainer CMD at
  Dockerfile line 166. Did not verify the flag existed in
  `run_phase2_eval.py` argparse. From this commit onward, the trainer
  image is broken.

- **2026-04-09 through 2026-04-20:** Runs 6, 7, 8 all use startup
  scripts, not Docker. Trainer image not built-and-run. Break
  undetected.

- **2026-04-20 (today):** KAN history investigation session ran
  `python scripts/run_phase2_eval.py --help 2>&1 | Select-String "skip"`.
  Output showed `[--skip-nn] [--skip-svm]` with no `[--skip-kan]`.
  Cross-referenced with Dockerfile line 166 grep which showed
  `--skip-kan` being passed. Mismatch identified.

- **2026-04-20 (commit 8f9eb60):** `--skip-kan` argparse flag added
  to `scripts/run_phase2_eval.py`. From this commit onward, the
  Dockerfile trainer CMD is valid. No Dockerfile change required.

## Root cause

The 671e48d commit modified the Dockerfile and the same-session plan
was to modify `run_phase2_eval.py` to honour `--skip-kan`. The
argparse change was deferred because the ROADMAP (lines 206-212) had
already specified it as a future task (item 4: "Add `--skip-kan`
flag as optional override (do not hardcode removal again)") and the
hardcoded `pop("kan", None)` (from commit a0a732d on 2026-04-05) made
the flag's absence from argparse functionally irrelevant — KAN was
going to be popped either way. The Dockerfile author anticipated the
flag would exist soon and wrote the CMD against the anticipated
interface.

Contributing factor: no CI job builds and runs the trainer image. CI
builds the API image (`--target api`, the default) and runs a smoke
test. The trainer image never enters the CI matrix, so argparse
errors at CMD time were never surfaced.

## Why undetected

Three converging reasons:

1. **No CI coverage of the trainer image.** The repository's CI
   workflow builds and smoke-tests the `api` stage only. The
   `trainer` stage is built-but-not-run by `docker build --target
   trainer`, but `docker build` does not execute `CMD`; CMD only
   runs at `docker run` time.

2. **No production trainer invocations.** All training happens on
   cloud VMs provisioned by startup scripts (`scripts/gcp_run6_startup.sh`
   and descendants) that `git clone` the repo and run
   `python scripts/run_phase2_eval.py` directly. The Docker trainer
   image is an artefact for hypothetical scheduled retraining jobs
   that have never been scheduled.

3. **Log inspection habits.** When the trainer CMD *would* have
   failed, it would have done so at the `docker run` entry point
   before any application logging. No one ran `docker run`, so no
   error was logged. The build succeeds (argparse is not evaluated at
   build time); it's only at run time that the error appears.

## Resolution

Commit 8f9eb60 (2026-04-20) added the `--skip-kan` argparse flag to
`scripts/run_phase2_eval.py`. The Dockerfile at line 166 now passes
a flag that argparse accepts. Default behaviour: when invoked with
`--skip-kan`, KAN is excluded from the ensemble; KANClassifier's
100K-sample subsample gate (commit 2389ee2) makes the CPU trainer's
behaviour well-defined even without the flag, but the Dockerfile
trainer CMD explicitly passes `--skip-kan` for belt-and-braces
caution on CPU-only hardware.

The fix requires no Dockerfile change and no rebuild. The next time
`docker build --target trainer -t genomic-variant-trainer:... .`
followed by `docker run` is attempted against origin/main at 8f9eb60
or later, it will succeed.

## Prevention

Two recommendations, filed here but deferred implementation:

1. **Add a trainer-image smoke test to CI.** Current CI builds and
   tests the API image. Extend with a `docker build --target
   trainer` followed by `docker run genomic-variant-trainer:ci-smoke
   --help` (or similar cheap invocation that exercises the CMD path
   without actually training). Would have caught this within minutes
   of commit 671e48d. Cost: ~30s additional CI time per PR.

2. **argparse-flag-matches-Dockerfile check in preflight.**
   `scripts/preflight_check.py` could grep the Dockerfile for flags
   passed to any Python script and verify each flag exists in that
   script's argparse. Cheap to implement, generalises beyond this
   specific case. Cost: negligible.

Neither is urgent. The failure mode is "trainer container exits
immediately with argparse error", not "silent bad training run". If
a future team member does attempt to build and run the trainer image,
the feedback loop is immediate.

## References

- Dockerfile line 166 (current state on main):
  `"--skip-nn", "--skip-svm", "--skip-kan", \`
- `scripts/run_phase2_eval.py` argparse at lines ~60-80 (post-8f9eb60):
  now includes `--skip-kan` argument.
- `docs/ROADMAP.md` lines 206-212: KAN Re-enablement Checklist (items
  3 and 4 are the specifications this incident's resolution fulfils).
- Commit 671e48d: Dockerfile change that introduced the mismatch.
- Commit 8f9eb60: argparse change that resolves the mismatch.
- Commit 2389ee2: upstream 100K subsample gate that makes KAN's
  Dockerfile trainer behaviour safe either way.

## Deferred related incident

`requirements.txt` pins `starlette==1.0.0` while
`prometheus-fastapi-instrumentator==7.1.0` (inherited via
`requirements-api.lock`) requires `starlette<1.0`. Pip emits a
non-fatal ERROR during CI install. Separate from this incident.
Per user instruction, filed after Run 9 completes.
