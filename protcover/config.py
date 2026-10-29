"""Configuration loading and repository paths."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(path)
    return cfg


def raw_dir(cfg):
    return ROOT / cfg["data"]["raw_dir"]


def processed_dir(cfg):
    d = ROOT / cfg["data"]["processed_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir():
    d = ROOT / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def predictor_names(cfg):
    return list(cfg["predictors"].keys())
