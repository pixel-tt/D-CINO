"""Train with the paper's free native-step objective and validation-only selection."""
import argparse
from pathlib import Path
import random
import signal
import time
import traceback
import numpy as np
import torch
from .io import checkpoint, configure, load_split, read_json, write_json, sha256
from .models.factory import METHODS, build_model
from .models.fusion import tensor_sha256
from .training_math import sequence_loss, rng_state, restore_rng


def sampling(labels, seed, updates, windows=(4, 8, 16), per_family=8):
    """Original rank sampler; eight draws per family and cyclic windows by default."""
    rng = np.random.default_rng(26101550 + seed)
    smooth = np.flatnonzero(labels == "smooth")
    other = np.flatnonzero(np.isin(labels, ("discontinuous", "dam")))
    if not len(smooth) or not len(other):
        raise ValueError("Balanced sampling requires both families")
    ranks = rng.random((updates, 2, per_family))
    identities = np.concatenate((smooth[(ranks[:, 0] * len(smooth)).astype(int)],
                                  other[(ranks[:, 1] * len(other)).astype(int)]), axis=1)
    return identities, np.resize(np.asarray(windows), updates), rng.bit_generator.state


@torch.inference_mode()
def validation_loss(model, sequence, device, steps=40):
    """Equal functions and all native times; fixed conservative channel scales."""
    was_training = model.training
    model.eval()
    per_step = np.zeros(steps, dtype=np.float64)
    for start in range(0, len(sequence), 8):
        samples = sequence[start:start + 8].to(device)
        state = samples[:, 0]
        for index in range(1, steps + 1):
            state, _ = model.step(state)
            per_step[index - 1] += float(((state - samples[:, index]) / model.scale).square().mean()) * len(samples) / len(sequence)
    model.train(was_training)
    return float(per_step.mean()), per_step.tolist()


def train(args):
    """Run or resume one fit; retain failures and never load the test split.

    AdamW acts on the folded coordinates. The raw minimum validation loss selects
    best.pt; a separate relative-improvement anchor controls plateau stopping.
    A completed or failed run cannot be reopened. SIGINT pauses with full state.
    """
    config = read_json(args.config)
    config.update(method=args.method, seed=args.seed, device=args.device, threads=args.threads)
    config["initialization_asset"] = None if args.initialization is None else args.initialization.name
    config["initialization_sha256"] = None if args.initialization is None else sha256(args.initialization)
    code_root = Path(__file__).resolve().parent
    config["code_sha256"] = {p.relative_to(code_root).as_posix(): sha256(p) for p in sorted(code_root.rglob("*.py"))}
    settings = config["training"]
    configure(args.threads)
    device, folder = torch.device(args.device), Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "status.json").exists() or (folder / "test-evaluated.json").exists():
        raise FileExistsError("Terminal or test-exposed runs cannot resume")
    if (folder / "config.json").exists() and not args.resume:
        raise FileExistsError("Existing run requires --resume")
    sequence, labels, train_ids, train_binding = load_split(args.data, "train", config)
    validation, _, val_ids, val_binding = load_split(args.data, "validation", config)
    if set(train_ids) & set(val_ids) or train_binding["sha256"] == val_binding["sha256"]:
        raise ValueError("Training and validation functions must be disjoint")
    if sequence.shape[1] <= max(settings["windows_steps"]) or validation.shape[1] <= settings["validation_steps"]:
        raise ValueError("Trajectory is shorter than objective/validation horizon")
    config["data"] = {"train": train_binding, "validation": val_binding}
    if args.resume and read_json(folder / "config.json") != config:
        raise ValueError("Resume configuration/data differs")
    write_json(folder / "config.json", config)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = build_model(config["problem"], args.method, args.seed, config["dt"], args.initialization).to(device)
    if not any(True for _ in model.parameters()):
        raise ValueError("Parameter-free carrier controls are evaluated without training")
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=settings["scheduler_factor"],
        patience=settings["scheduler_patience"], threshold=settings["relative_threshold"], threshold_mode="rel", min_lr=settings["min_lr"])
    identities, endpoints, sampling_rng = sampling(labels, args.seed, settings["max_updates"],
        settings["windows_steps"], settings["batch_size"] // 2)
    progress = {"update": 0, "best": None, "anchor": None, "stale": 0, "best_update": 0,
                "history": [], "trace": [], "transition_evaluations": 0, "elapsed_seconds": 0.}
    if args.resume:
        saved = torch.load(folder / "last.pt", map_location=device, weights_only=False)
        if saved["config"] != config:
            raise ValueError("Resume checkpoint configuration differs")
        if sha256(folder / "best.pt") != saved["best_checkpoint_sha256"]:
            raise ValueError("Selected checkpoint changed after pause")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        progress = saved["progress"]
        restore_rng(saved["rng"])
    else:
        checkpoint(folder / "initialization.pt", {"model": model.state_dict(), "seed": args.seed,
            "optimizer_updates": 0, "tensor_sha256": tensor_sha256(model.state_dict()), "rng": rng_state(), "config": config})
        np.savez_compressed(folder / "sampling.npz", identities=identities, endpoints=endpoints)
        write_json(folder / "environment.json", {"torch": str(torch.__version__), "numpy": np.__version__, "device": str(device),
            "threads": args.threads, "amp": False, "tf32": False, "sampler_rng": sampling_rng})
    requested, previous = [], {}
    for number in (signal.SIGINT, signal.SIGTERM):
        previous[number] = signal.getsignal(number)
        signal.signal(number, lambda signum, _frame: requested.append(signum))
    started, prior = time.perf_counter(), progress["elapsed_seconds"]

    def select(value):
        checkpoint(folder / "best.pt", {"model": model.state_dict(), "update": progress["update"],
            "validation_loss": value, "config": config, "tensor_sha256": tensor_sha256(model.state_dict())})

    def save():
        from .io import sha256
        progress["elapsed_seconds"] = prior + time.perf_counter() - started
        checkpoint(folder / "last.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "rng": rng_state(), "progress": progress, "config": config,
            "best_checkpoint_sha256": sha256(folder / "best.pt") if (folder / "best.pt").exists() else None})
        write_json(folder / "history.json", progress["history"])
        write_json(folder / "trace.json", progress["trace"])

    state, reason, code = "completed", "maximum_updates", 0
    try:
        if not args.resume:
            value, components = validation_loss(model, validation, device, settings["validation_steps"])
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite initial validation")
            progress.update(best=value, anchor=value)
            progress["history"].append({"update": 0, "loss": value, "components": components})
            select(value)
            save()
        stopped = progress["update"] >= settings["min_updates"] and progress["stale"] >= settings["stop_patience"]
        if stopped:
            reason = "validation_plateau"
        while progress["update"] < settings["max_updates"] and not stopped:
            if requested or (args.pause_after is not None and progress["update"] >= args.pause_after):
                save()
                write_json(folder / "pause.json", {"state": "paused", "update": progress["update"]})
                return 75
            if prior + time.perf_counter() - started > settings["max_seconds"]:
                raise TimeoutError("Training wall-time ceiling exceeded")
            index = progress["update"]
            optimizer.zero_grad(set_to_none=True)
            loss = sequence_loss(model, sequence[identities[index]].to(device), int(endpoints[index]))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["clip"], error_if_nonfinite=True)
            optimizer.step()
            progress["update"] += 1
            progress["transition_evaluations"] += int(endpoints[index])
            update = progress["update"]
            if update == 1 or update % 100 == 0:
                progress["trace"].append({"update": update, "train_loss": float(loss.detach()), "gradient_norm": float(norm), "lr": optimizer.param_groups[0]["lr"]})
            if update % settings["validation_interval"] == 0 or update == settings["max_updates"]:
                value, components = validation_loss(model, validation, device, settings["validation_steps"])
                if not np.isfinite(value):
                    raise FloatingPointError("Nonfinite validation")
                if value < progress["best"]:
                    progress["best"], progress["best_update"] = value, update
                    select(value)
                if value < progress["anchor"] * (1 - settings["relative_threshold"]):
                    progress["anchor"], progress["stale"] = value, 0
                else:
                    progress["stale"] += 1
                scheduler.step(value)
                progress["history"].append({"update": update, "loss": value, "components": components,
                    "best": progress["best"], "stale": progress["stale"], "lr": optimizer.param_groups[0]["lr"]})
                print("VALIDATION", config["problem"], args.method, update, value, flush=True)
                stopped = update >= settings["min_updates"] and progress["stale"] >= settings["stop_patience"]
                if stopped:
                    reason = "validation_plateau"
            if update % 100 == 0 or stopped or update == settings["max_updates"]:
                save()
        save()
    except Exception as error:
        numerical = isinstance(error, FloatingPointError) or any(word in str(error).lower() for word in ("nonfinite", "non-finite"))
        state, code = ("numerical_failure", 2) if numerical else ("infrastructure_failure", 3)
        reason = f"{type(error).__name__}: {error}"
        progress["elapsed_seconds"] = prior + time.perf_counter() - started
        write_json(folder / "failure.json", {"error": reason, "traceback": traceback.format_exc()})
        checkpoint(folder / "failure-current.pt", {"model": model.state_dict(), "resumable": False})
        write_json(folder / "history.json", progress["history"])
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    write_json(folder / "status.json", {"terminal": True, "state": state, "reason": reason,
        "test_accessed": False, "best_update": progress["best_update"], "best_validation_loss": progress["best"],
        "optimizer_updates": progress["update"], "transition_evaluations": progress["transition_evaluations"],
        "training_seconds": progress["elapsed_seconds"],
        "selected_checkpoint_sha256": sha256(folder / "best.pt") if (folder / "best.pt").exists() else None,
        "convergence_status": "not_certified", "profile": config["profile"]})
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS[:-1], default="dcino")
    parser.add_argument("--seed", type=int, default=1621)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--initialization", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pause-after", type=int)
    return train(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
