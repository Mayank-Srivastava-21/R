import os
import json
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _json_default(o):
    """Never let a numpy/torch object kill a save (and thus a multi-hour run)."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().numpy().tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
