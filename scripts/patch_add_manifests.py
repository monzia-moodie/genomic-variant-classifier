import re
from pathlib import Path

MANIFEST_FUNC = '''
def _write_model_manifest(artifact_path):
    """Write a JSON manifest recording the library versions used to create this artifact."""
    import json, platform, importlib.metadata
    from datetime import datetime, timezone
    artifact_path = Path(artifact_path)
    libraries = [
        "numpy", "scikit-learn", "catboost", "lightgbm",
        "xgboost", "joblib", "pandas", "scipy",
    ]
    manifest = {
        "artifact":   artifact_path.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python":     platform.python_version(),
        "platform":   platform.platform(),
        "libraries":  {
            lib: importlib.metadata.version(lib)
            for lib in libraries
        },
    }
    manifest_path = artifact_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path
'''

# ── 1. InferencePipeline.save() ──────────────────────────────────────────────
pipeline_path = Path("src/genomic_variant_classifier/api/pipeline.py")
pipeline_src = pipeline_path.read_text(encoding="utf-8")

old = (
    "    def save(self, path: str | Path) -> None:\n"
    "        import joblib\n"
    "        path = Path(path)\n"
    "        path.parent.mkdir(parents=True, exist_ok=True)\n"
    "        joblib.dump(self, path)\n"
    "        logger.info(\"InferencePipeline saved \u2192 %s\", path)"
)
new = (
    "    def save(self, path: str | Path) -> None:\n"
    "        import joblib\n"
    "        path = Path(path)\n"
    "        path.parent.mkdir(parents=True, exist_ok=True)\n"
    "        joblib.dump(self, path)\n"
    "        _write_model_manifest(path)\n"
    "        logger.info(\"InferencePipeline saved \u2192 %s\", path)"
)
assert old in pipeline_src, "InferencePipeline.save() block not found — check line numbers"
pipeline_src = pipeline_src.replace(old, new, 1)

# Inject helper before the class definition
inject_before = "class InferencePipeline:"
assert inject_before in pipeline_src, "InferencePipeline class not found"
pipeline_src = pipeline_src.replace(inject_before, MANIFEST_FUNC + "\n" + inject_before, 1)
pipeline_path.write_text(pipeline_src, encoding="utf-8")
print("patched: src/api/pipeline.py")

# ── 2. VariantEnsemble.save() ────────────────────────────────────────────────
ensemble_path = Path("src/genomic_variant_classifier/models/variant_ensemble.py")
ensemble_src = ensemble_path.read_text(encoding="utf-8")

old = (
    "    def save(self, path: Optional[Path] = None) -> None:\n"
    "        import joblib\n"
    "        path = Path(path or self.config.model_dir / \"ensemble.joblib\")\n"
    "        path.parent.mkdir(parents=True, exist_ok=True)\n"
    "        joblib.dump(self, path)\n"
    "        logger.info(\"Ensemble saved to %s\", path)"
)
new = (
    "    def save(self, path: Optional[Path] = None) -> None:\n"
    "        import joblib\n"
    "        path = Path(path or self.config.model_dir / \"ensemble.joblib\")\n"
    "        path.parent.mkdir(parents=True, exist_ok=True)\n"
    "        joblib.dump(self, path)\n"
    "        _write_model_manifest(path)\n"
    "        logger.info(\"Ensemble saved to %s\", path)"
)
assert old in ensemble_src, "VariantEnsemble.save() block not found — check line numbers"
ensemble_src = ensemble_src.replace(old, new, 1)

inject_before = "class VariantEnsemble:"
assert inject_before in ensemble_src, "VariantEnsemble class not found"
ensemble_src = ensemble_src.replace(inject_before, MANIFEST_FUNC + "\n" + inject_before, 1)
ensemble_path.write_text(ensemble_src, encoding="utf-8")
print("patched: src/models/variant_ensemble.py")

# ── 3. scripts/run_phase2_eval.py — scaler + gnn artifacts ──────────────────
script_path = Path("scripts/run_phase2_eval.py")
script_src = script_path.read_text(encoding="utf-8")

old = (
    "        import joblib\n"
    "        joblib.dump(prep.scaler, outdir / \"scaler.joblib\")\n"
    "        logger.info(\"Scaler saved to %s/scaler.joblib\", outdir)"
)
new = (
    "        import joblib\n"
    "        joblib.dump(prep.scaler, outdir / \"scaler.joblib\")\n"
    "        _write_model_manifest(outdir / \"scaler.joblib\")\n"
    "        logger.info(\"Scaler saved to %s/scaler.joblib\", outdir)"
)
if old in script_src:
    script_src = script_src.replace(old, new, 1)
    print("patched: scripts/run_phase2_eval.py (scaler)")
else:
    print("WARNING: scaler dump block not matched in run_phase2_eval.py — skipping")

old2 = 'joblib.dump(gnn_model, outdir / "models" / "gnn_model.joblib")'
new2 = (
    'joblib.dump(gnn_model, outdir / "models" / "gnn_model.joblib")\n'
    '                    _write_model_manifest(outdir / "models" / "gnn_model.joblib")'
)
if old2 in script_src:
    script_src = script_src.replace(old2, new2, 1)
    print("patched: scripts/run_phase2_eval.py (gnn_model)")

old3 = 'joblib.dump(gnn_scorer, outdir / "models" / "gnn_scorer.joblib")'
new3 = (
    'joblib.dump(gnn_scorer, outdir / "models" / "gnn_scorer.joblib")\n'
    '                    _write_model_manifest(outdir / "models" / "gnn_scorer.joblib")'
)
if old3 in script_src:
    script_src = script_src.replace(old3, new3, 1)
    print("patched: scripts/run_phase2_eval.py (gnn_scorer)")

# Inject helper near top of script (after imports)
if "_write_model_manifest" not in script_src:
    inject_after = "logger = logging.getLogger(__name__)"
    if inject_after in script_src:
        script_src = script_src.replace(
            inject_after,
            inject_after + "\n" + MANIFEST_FUNC,
            1,
        )
script_path.write_text(script_src, encoding="utf-8")
print("patched: scripts/run_phase2_eval.py")

print("\nAll done. Run the test suite to verify nothing broke.")
