"""Measure native-step and complete-forecast latency in alternating cohort order."""
import argparse
from pathlib import Path
import time
import platform
import numpy as np
import torch
from .io import configure, load_split, read_json, write_json
from .models.factory import build_model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="*", default=[], help="Completed fits of one system; carrier-only is included once")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--function", type=int, help="Default: first smooth general-test function")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--steps", type=int, default=40, help="Complete forecast horizon, independent of step timing")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0 or args.steps < 1:
        raise ValueError("Invalid timing budget")
    config = read_json(args.config)
    configure(args.threads)
    truth, labels, ids, _ = load_split(args.data, "test", config)
    index = labels.tolist().index("smooth") if args.function is None else args.function
    initial = truth[index:index + 1, 0].to(args.device)
    if not len(initial):
        raise ValueError("Function index outside test population")
    cohort = [("carrier_only", None, build_model(config["problem"], "carrier_only"))]
    for folder in args.run:
        if read_json(folder / "status.json")["state"] != "completed":
            raise ValueError("Timing requires completed fits")
        payload = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
        own = payload["config"]
        for key in ("problem", "cells", "dt", "profile"):
            if own[key] != config[key]:
                raise ValueError("Timing cohort configuration differs")
        method, seed = own["method"], own["seed"]
        model = build_model(config["problem"], method, seed, config["dt"])
        model.load_state_dict(payload["model"], strict=True)
        cohort.append((method, seed, model))
    records = []
    for method, seed, model in cohort:
        records.append({"method": method, "seed": seed, "model": model.to(args.device).eval(),
            "fit_id": f"{method}:{seed}", "step_seconds": [], "forecast_seconds": [], "failures": []})
    if len({row["fit_id"] for row in records}) != len(records):
        raise ValueError("Duplicate timing fit identity")

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    def advance(row):
        candidate, _ = row["model"].step(initial)
        return bool(torch.isfinite(candidate).all())

    def forecast(row):
        model, method = row["model"], row["method"]
        if config["problem"] == "swe2d":
            from .models.two_dimensional.metrics import predict
            _, _, _, failed, _ = predict(model, initial, args.steps, wrapped=method not in ("fno", "unet"))
            return not bool(failed.any())
        if method in ("fno", "unet", "fno_physical_wrapper"):
            from .models.one_dimensional.metrics import forecast as predict
            from .models.one_dimensional import models_core
            output = predict(model, initial, args.steps, method, models_core)
            return not bool(output["numerical_failure_native"].any())
        from .models.one_dimensional.forecast import rollout
        _, _, failed, _ = rollout(model, initial, args.steps, config["dt"], frame="moving")
        return not bool(failed.any())

    for row in records:
        for _ in range(args.warmups):
            for arm, callback in (("step", advance), ("forecast", forecast)):
                if not callback(row):
                    row["failures"].append({"arm": arm, "phase": "warmup"})
    sync()
    orders = []
    for repeat in range(args.repeats):
        order = records if repeat % 2 == 0 else list(reversed(records))
        orders.append([row["fit_id"] for row in order])
        for row in order:
            for arm, callback in (("step", advance), ("forecast", forecast)):
                sync()
                started = time.perf_counter()
                good = callback(row)
                sync()
                row[arm + "_seconds"].append(time.perf_counter() - started)
                if not good:
                    row["failures"].append({"arm": arm, "phase": "measurement", "repeat": repeat})
    for row in records:
        del row["model"]
        for arm in ("step", "forecast"):
            samples = np.asarray(row[arm + "_seconds"])
            eligible = len(samples) == args.repeats and not row["failures"]
            row[arm + "_summary"] = {"eligible_complete_horizon": eligible,
                "median_seconds": float(np.median(samples)) if eligible else None,
                "q25_seconds": float(np.quantile(samples, .25)) if eligible else None,
                "q75_seconds": float(np.quantile(samples, .75)) if eligible else None}
    write_json(args.output, {"results": records, "measurement_orders": orders, "function_id": ids[index],
        "steps": args.steps, "warmups": args.warmups, "repetitions": args.repeats,
        "environment": {"torch": str(torch.__version__), "device": args.device, "threads": args.threads,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None, "python": platform.python_version()},
        "scope": "actual dispatchers; includes finite checks, frame, Q, output storage and dispatcher diagnostics; excludes metric reduction, loading, IO, reference generation and transfers",
        "isolation_requirement": "Run without concurrent training or other device inference; not programmatically enforced"})


if __name__ == "__main__":
    main()
