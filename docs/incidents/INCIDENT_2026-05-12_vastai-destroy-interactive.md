# INCIDENT 2026-05-12 — vastai destroy interactive prompt breaks auto-destroy

## Status

ROOT CAUSE IDENTIFIED — RESOLUTION DEFERRED TO PHASE 1.5
(`scripts/launch_run9_vm.sh` patch — pending source-of-truth review of
the current file). Manual workaround in place: destroy via web console
at https://cloud.vast.ai/instances/ until the launch script is patched.

## Summary

`vastai destroy instance <ID>` in CLI version ≥ 1.0.12 prompts
`Are you sure? [y/N]` and waits on stdin. When called from within the
`cleanup_if_setup_failed` trap function in `scripts/launch_run9_vm.sh`
(non-interactive shell, no stdin available), the call hangs
indefinitely or returns immediately with a non-zero exit and no
destruction occurs. The auto-destroy cost-safety mechanism is broken in
both pre-training failure mode (preflight) and post-training manual
mode (when invoked from local shell).

In Run 9 specifically, this manifested as: training crashed in
`ensemble.save()`; the script's trap printed the manual-destroy
reminder; Monzia ran `vastai destroy instance 36588175` from local
PowerShell; the call hung. Manual destruction had to be performed via
the Vast.ai web console after a wait period during which the instance
continued to bill. Estimated idle cost: ~$4.30 (~9 h × $0.473/hr).

## Cross-references

- `docs/sessions/SESSION_2026-05-12.md` §Costs and §Failed — recorded
  the manual web-console destruction.
- `docs/CHANGELOG.md` 2026-05-12 entry — *"Instance destroyed manually
  (auto-destroy broken in vastai 1.0.12)"*.
- `docs/incidents/INCIDENT_2026-05-12_launch-path-inconsistency.md` —
  same launch script, different bug class.
- Memory entry: *"vastai 1.0.12 destroy is interactive — `echo y |
  vastai destroy instance ID` for non-interactive"*.

## Evidence

### Source of the bug

`scripts/launch_run9_vm.sh` lines 36–37 (file content reviewed in
chat session 2026-05-13):

```bash
vastai destroy instance "$INSTANCE_ID" 2>&1 || \
    echo "[auto-destroy] CLI destroy FAILED -- destroy from web console NOW: https://cloud.vast.ai/instances/"
```

The call does not pipe `y` to stdin and does not pass any
non-interactive flag (`--yes`, `--no-prompt`, etc., depending on which
CLI version is installed).

### Behaviour in non-interactive shells

In `cleanup_if_setup_failed`, the trap function runs in the same shell
as the script. `nohup` (used in Run 9 launch per session doc lessons)
redirects stdin from `/dev/null`. When `vastai destroy` reads from
stdin, it either:
- Hangs forever waiting for input (older CLI behaviour)
- Receives EOF and exits with `OSError: Bad file descriptor` (CLI 1.0.12 behaviour per session doc)

Either failure mode means the instance is not destroyed.

## Root cause

`vastai-cli` version 1.0.12 (released some time before 2026-05-12)
added an interactive confirmation step to the `destroy` subcommand.
This was a breaking change for any automation that relied on the
unattended pre-1.0.12 behaviour. The `launch_run9_vm.sh` script was
written before this CLI change and was never updated.

## Remediation

### Quick fix (Phase 1.5)

Pipe `y` to stdin of the destroy call:

```bash
echo y | vastai destroy instance "$INSTANCE_ID" 2>&1 || \
    echo "[auto-destroy] CLI destroy FAILED -- destroy from web console NOW: https://cloud.vast.ai/instances/"
```

If the CLI version supports an explicit non-interactive flag
(`--yes`, `--no-prompt`), prefer that over `echo y |` for robustness
against future prompt-text changes. Probe at launch time:

```bash
if vastai destroy --help 2>&1 | grep -q -- --yes; then
    DESTROY_ARGS="--yes"
else
    DESTROY_ARGS=""
fi
# ...
${DESTROY_ARGS:+vastai destroy $DESTROY_ARGS} || echo y | vastai destroy
```

### Companion fixes in the same Phase 1.5 patch

Per chat-side rigorous evaluation of `scripts/launch_run9_vm.sh` on
2026-05-13, several other defects compound this bug's failure mode:

- **C2 (ALL DONE prints on failure):** after an ablation `break`, the
  script falls through to `echo "ALL DONE"` and exits 0; the trap then
  reads `rc=0` and prints nothing. Even with the `echo y` fix, the
  trap doesn't fire on training failures because the exit code was
  swallowed by `break`. Fix: track `LOOP_FAILED=yes`, explicit
  `exit 1` after the loop.
- **H1 (silent venv activation):** `source /venv/main/bin/activate
  2>/dev/null || true` masks activation failures and silently falls
  through to the system Python. Fix: explicit FATAL on activation
  failure when /venv/main exists.
- **H3 (vastai CLI not guaranteed on the image):** the Vast.ai 2026
  PyTorch image does not include `vastai` by default; the trap silently
  falls through to "DESTROY MANUALLY" if the CLI is missing. Fix: install
  vastai CLI as part of the bootstrap (`pip install vastai`) before
  training begins.
- **H2 (no env var validation):** `INSTANCE_ID` is documented as
  required but never checked at script entry. Fix: `: "${INSTANCE_ID:?required}"`.
- **H4 (no HEAD-sha assertion):** the script prints `git log -1` but
  doesn't verify HEAD matches what the operator intended to launch. Fix:
  accept optional `EXPECTED_HEAD_SHA` env var, abort on mismatch.

A unified patch addressing C1 (this INCIDENT) + C2 + H1–H4 will be
drafted as a single follow-up bundle once the current
`scripts/launch_run9_vm.sh` source is verified one more time post any
in-flight commits.

### Workaround until patch lands

For any Vast.ai run launched before the patch ships, the operator
must destroy the instance via the web console
(https://cloud.vast.ai/instances/) within minutes of training
completion or failure. Memory entry: *"Vast.ai workflow: SCP up →
train → SCP back → destroy immediately."*

## Lessons

- **External CLI tools can introduce breaking changes silently.** Any
  shell script that depends on a third-party CLI should pin the CLI
  version (or feature-test at runtime via `--help | grep -q`) and
  defensively pipe `y` to any prompt-capable subcommand.
- **Cost-safety nets must be tested non-interactively.** The trap
  function in `launch_run9_vm.sh` was never exercised end-to-end with
  a real INSTANCE_ID before Run 9. A no-op probe (e.g., destroy on a
  test instance immediately after creation) would have caught the
  hang at first use, not at the worst possible moment.
- **`echo y | <command>` is the bare-minimum compatibility shim.** Use
  for any CLI command that gained a prompt in a recent release. Add
  a comment with the release note URL.

## Sign-off

INCIDENT moves to RESOLVED when:
- `scripts/launch_run9_vm.sh` is patched (Phase 1.5 unified patch)
- A test launch + immediate self-destroy on a 1-minute test instance
  confirms the trap function destroys the instance without manual
  intervention.
