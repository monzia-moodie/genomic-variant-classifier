import json, platform, importlib.metadata
from datetime import datetime, timezone
from pathlib import Path

def write_model_manifest(artifact_path):
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
    print(f"Written: {manifest_path}")
    print(json.dumps(manifest, indent=2))
    return manifest_path

write_model_manifest("models/phase4_pipeline.joblib")
write_model_manifest("models/phase4_pipeline_calibrated.joblib")
write_model_manifest("models/phase2_pipeline.joblib")
