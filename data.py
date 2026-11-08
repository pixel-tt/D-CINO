"""Generate independent paper or small smoke trajectories with the original solver."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .io import configure, read_json, write_json


def generate_1d(config, split):
    """Preserve original short-train batches and the 32-function continuation.

    Batch membership matters because the reference solver uses one CFL step for
    the entire batch. The production training recipe is 96 original functions
    plus 976 additional discontinuous functions, followed by a t=.1 continuation.
    Validation uses the fresh 24 functions; general test uses 32 fresh functions.
    """
    from .models.one_dimensional import solver
    problem, recipe = config["problem"], config["data"][split]
    fine, cells, dt = recipe["reference_cells"], config["cells"], config["dt"]
    seed = recipe["seed"] + (7919 if problem == "euler" else 0)
    times = np.arange(recipe["steps"] + 1) * dt * recipe.get("output_stride", 1)
    records = []
    if config["profile"] == "paper" and split == "train":
        base, labels = solver.initial_conditions(96, fine, 104729 + (7919 if problem == "euler" else 0), problem)
        short, info = solver.solve(base, problem, np.arange(1, 6) * dt)
        records.append(info)
        extra, extra_labels = solver.initial_conditions(1952, fine, seed, problem)
        extra = extra[1::2]
        pieces = [short]
        for start in range(0, len(extra), 128):
            value, info = solver.solve(extra[start:start + 128], problem, np.arange(1, 6) * dt)
            pieces.append(value)
            records.append(info)
        original = np.concatenate(pieces)
        labels = np.concatenate((labels, extra_labels[1::2]))
        coarse = []
        for start in range(0, len(original), 32):
            continuation, info = solver.solve(original[start:start + 32, 5], problem, np.arange(1, 36) * dt)
            combined = np.concatenate((original[start:start + 32], continuation[:, 1:]), axis=1)
            coarse.append(solver.restrict(combined, cells).transpose(0, 1, 3, 2).copy())
            records.append(info)
            print("GENERATED", problem, split, min(start + 32, len(original)), len(original), flush=True)
        states = np.concatenate(coarse)
        # The original container's single seed referred to the added discontinuous pool.
        ids = np.asarray([f"base:{104729 + (7919 if problem == 'euler' else 0)}:{i}" for i in range(96)]
                         + [f"extra:{seed}:{i}" for i in range(1, 1952, 2)])
    else:
        count = recipe["count"]
        initial, labels = solver.initial_conditions(2 * count if recipe.get("smooth_only") else count, fine, seed, problem)
        selected = np.arange(0, 2 * count, 2) if recipe.get("smooth_only") else np.arange(count)
        initial, labels = initial[selected], labels[selected]
        if "amplitude" in recipe:
            mean = initial.mean(1, keepdims=True)
            initial = mean + recipe["amplitude"] * (initial - mean)
        values, info = solver.solve(initial, problem, times[1:])
        states = solver.restrict(values, cells).transpose(0, 1, 3, 2)
        records.append(info)
        ids = np.asarray([f"{seed}:{i}" for i in selected])
    # Training/validation runtime archives were compact float32; truth stays float64.
    return {"states": states.astype(np.float32 if split in ("train", "validation") else np.float64),
            "labels": labels, "ids": ids, "times": times, "seed": seed}, records


def generate_2d(config, split, device):
    """Master quadrature, fixed solver batches, unsplit float64 integration."""
    from .data_math import initial, parameters
    from .models.two_dimensional.swe2d import solve, restrict
    recipe = config["data"][split]
    items = parameters(recipe["count"], recipe["seed"])
    times = np.arange(recipe["steps"] + 1) * config["dt"]
    arrays, records = [], []
    for start in range(0, len(items), recipe["solver_batch"]):
        group = items[start:start + recipe["solver_batch"]]
        state = initial(group, recipe["reference_cells"], amplitude=recipe.get("amplitude", 1.), master=recipe["master_cells"], device=device)
        trajectory, info = solve(state, times[1:])
        arrays.append(restrict(trajectory, config["cells"]).cpu().numpy())
        records.append(info)
        print("GENERATED", "swe2d", split, start + len(group), len(items), flush=True)
    return {"states": np.concatenate(arrays).astype(np.float32 if split in ("train", "validation") else np.float64),
            "labels": np.asarray([p["family"] for p in items]), "ids": np.asarray([p["id"] for p in items]), "times": times}, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test", "weak", "veryweak"), nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="2D reference arithmetic device")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    config = read_json(args.config)
    configure(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    for split in args.split:
        target = args.output / (split + ".npz")
        if target.exists():
            raise FileExistsError("Preserve existing generated data: " + target.name)
        arrays, records = (generate_2d(config, split, args.device) if config["problem"] == "swe2d"
                           else generate_1d(config, split))
        np.savez_compressed(target, **arrays)
        write_json(target.with_suffix(".json"), {"profile": config["profile"], "split": split,
            "problem": config["problem"], "shape": list(arrays["states"].shape), "recipe": config["data"][split],
            "reference_device": args.device if config["problem"] == "swe2d" else "numpy-cpu", "solver_diagnostics": records,
            "historical_archive_identity": "not asserted; generated NPZ includes portable IDs and metadata"})


if __name__ == "__main__":
    main()
