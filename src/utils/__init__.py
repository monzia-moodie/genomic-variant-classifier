"""Utility functions."""
import logging
import random
import numpy as np
from pathlib import Path
import yaml

def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)
