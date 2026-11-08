"""Small portable configuration, split and checkpoint helpers."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import torch


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def configure(threads):
    if threads < 1:
        raise ValueError("Need positive CPU thread count")
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_split(root, split, config, dtype=np.float32):
    """Read one explicit split only; accept exported states or bound paper NPZ.

    A data-map.json can name original arrays (n128/n64/states), layouts and
    hashes without copying large archives. Paths are relative to its directory.
    The train entry point calls this function only for train and validation.
    """
    root = Path(root)
    mapping = read_json(root / "data-map.json") if (root / "data-map.json").exists() else {}
    item = mapping.get(config["problem"], mapping).get(split, {"path": split + ".npz", "array": "states", "layout": "channels_first"})
    path = root / item["path"]
    if item.get("sha256") and sha256(path) != item["sha256"]:
        raise ValueError("Data archive checksum mismatch: " + path.name)
    with np.load(path, allow_pickle=False) as archive:
        states = archive[item.get("array", "states")]
        if item.get("layout", "channels_first") == "channels_last":
            if config["problem"] == "swe2d":
                raise ValueError("2D data must be channels-first")
            states = states.transpose(0, 1, 3, 2)
        # Restrict a supplied fine 1D reference by conservative block averages.
        if config["problem"] != "swe2d" and states.shape[-1] != config["cells"]:
            factor, remainder = divmod(states.shape[-1], config["cells"])
            if remainder or factor < 1:
                raise ValueError("Fine grid must be divisible by observation grid")
            states = states.reshape(*states.shape[:-1], config["cells"], factor).mean(-1)
        labels = archive["labels"].astype(str) if "labels" in archive else np.full(len(states), item.get("family", "smooth"))
        ids = (archive["ids"].astype(str) if "ids" in archive
               else np.asarray([item.get("id_prefix", split) + ":" + str(i) for i in range(len(states))]))
        times = (archive["times"].astype(np.float64) if "times" in archive
                 else np.arange(states.shape[1]) * config["dt"] * item.get("time_stride", 1))
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate function identities")
    labels = np.asarray([{"smooth_nonparallel": "smooth", "circular_dam": "dam"}.get(x, x) for x in labels])
    if len(states) != len(ids) or len(states) != len(labels) or not np.isfinite(states).all():
        raise ValueError("Inconsistent/nonfinite trajectory data")
    channels = 2 if config["problem"] == "swe" else 3
    spatial = (config["cells"], config["cells"]) if config["problem"] == "swe2d" else (config["cells"],)
    if tuple(states.shape[2:]) != (channels, *spatial):
        raise ValueError("State channel/grid dimensions do not match the configuration")
    if config["profile"] == "paper":
        recipe = config["data"][split]
        if states.shape[:2] != (recipe["count"], recipe["steps"] + 1):
            raise ValueError("Paper profile requires its exact function count and output horizon")
        if split == "train" and config["problem"] != "swe2d":
            expected_smooth = 48
        else:
            expected_smooth = recipe["count"] if recipe.get("smooth_only") else recipe["count"] // 2
        if int((labels == "smooth").sum()) != expected_smooth:
            raise ValueError("Paper profile family composition differs from the bound recipe")
    return torch.from_numpy(states.astype(dtype)), labels, ids, {"sha256": sha256(path), "array": item.get("array", "states"), "shape": list(states.shape), "times": times.tolist()}
