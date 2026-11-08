"""Score a terminal validation-selected fit with the original failure policies."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from .io import configure, load_split, read_json, sha256, write_json
from .models.factory import METHODS, build_model


def evaluate(args):
    """Retain per-function native-time metrics, masks and full conservative fields.

    Native FNO/U-Net use physical autoregression; D-CINO, carrier-only and
    residual FNO use the canonical trajectory and deferred observations. The
    physical arm changes only the forecast frame. Failed functions remain in
    every mean with the inherited 1e6 penalty; centered errors are diagnostics.
    """
    config = read_json(args.config)
    configure(args.threads)
    device = torch.device(args.device)
    method, seed = args.method, args.seed
    if args.run is not None:
        status = read_json(args.run / "status.json")
        if status.get("state") != "completed":
            raise ValueError("Evaluate only a completed training terminal; failed fits must be reported as unavailable")
        saved = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
        if status.get("selected_checkpoint_sha256") and sha256(args.run / "best.pt") != status["selected_checkpoint_sha256"]:
            raise ValueError("Validation-selected checkpoint changed after training")
        fit_config = saved["config"]
        if saved["update"] != status["best_update"]:
            raise ValueError("Checkpoint was not the validation-selected state")
        for name in ("problem", "cells", "dt", "profile"):
            if fit_config[name] != config[name]:
                raise ValueError("Evaluation configuration differs: " + name)
        method, seed = fit_config["method"], fit_config["seed"]
    elif method not in ("carrier_only", "raw_carrier"):
        raise ValueError("A learned method requires --run; random weights are not an evaluation fit")
    model = build_model(config["problem"], method, seed, config["dt"])
    if args.run is not None:
        model.load_state_dict(saved["model"], strict=True)
    model = model.to(device).eval()
    truth_cpu, labels, ids, binding = load_split(args.data, args.split, config, dtype=np.float64)
    times = np.asarray(binding["times"])
    match = np.rint(times / config["dt"]).astype(int)
    if not np.allclose(match * config["dt"], times, atol=1e-12, rtol=0):
        raise ValueError("Reference times must be native-step multiples")
    steps = int(match[-1])
    initial = truth_cpu[:, 0].to(device=device, dtype=torch.float32)
    frame = args.frame
    if frame == "default":
        frame = "physical" if method in ("fno", "unet") else "moving"
    if method in ("fno", "unet", "fno_physical_wrapper") and args.frame != "default":
        raise ValueError("The published baseline frame is fixed; use --frame default")
    started = time.perf_counter()
    if config["problem"] == "swe2d":
        from .models.two_dimensional.metrics import predict, metrics
        prediction, _, theta, failed, failures = predict(model, initial, steps, wrapped=frame == "moving")
        truth = truth_cpu.to(device)
        values = metrics(prediction[:, match], truth, failed[:, match], initial_reference=truth[:, :1])
        arrays = {"prediction": prediction[:, match].cpu().numpy(), "truth": truth_cpu.numpy(),
                  "theta": theta.cpu().numpy(), "failure_mask": failed.cpu().numpy()}
        primary = "fluctuation_l2" if args.split == "weak" else "state_rms"
    else:
        from .models.one_dimensional import models_core
        from .models.one_dimensional.metrics import forecast as baseline_forecast, metrics, physical_diagnostics
        from .models.one_dimensional.forecast import forecast
        if method in ("fno", "unet", "fno_physical_wrapper"):
            result = baseline_forecast(model, initial, steps, method, models_core)
        elif method == "raw_carrier":
            # Reuse the actual current raw-carrier diagnostic, without positivity rejection.
            from .models.one_dimensional.raw_forecast import raw_forecast_1d
            result = raw_forecast_1d(model, initial, steps, frame)
        else:
            result = forecast(model, initial, steps, frame=frame)
        prediction, failed = result["prediction_native"], result["numerical_failure_native"]
        values, denominators = metrics(prediction[:, match], truth_cpu.numpy(), config["problem"], failed[:, match])
        invalid_native = physical_diagnostics(prediction, config["problem"])["physical_invalid"]
        values["physical_invalid_mask"] = invalid_native[:, match]
        values["denominators"] = denominators
        failures = result["failures"]
        arrays = {"prediction": prediction[:, match], "truth": truth_cpu.numpy(), "theta": result["theta"],
                  "failure_mask": failed, "physical_invalid_mask_native": invalid_native}
        primary = ("weak_physical_RMS_over_endpoint_truth_fluctuation" if args.split in ("weak", "veryweak")
                   else "S_scaled_state_RMS")
    seconds = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=False)
    arrays.update(times=times, native_times=np.arange(steps + 1) * config["dt"], labels=labels, ids=ids)
    np.savez_compressed(args.output / "predictions.npz", **arrays)
    from .models.one_dimensional.metrics import safe_json
    metric_values = values[primary]
    summary = {"profile": config["profile"], "problem": config["problem"], "method": method,
        "seed": None if method in ("carrier_only", "raw_carrier") else seed, "frame": frame,
        "split": args.split, "data": binding, "function_ids": ids.tolist(), "families": labels.tolist(),
        "primary_metric": primary, "metrics": safe_json(values), "failures": safe_json(failures),
        "endpoint_function_mean": float(np.asarray(metric_values)[:, -1].mean()),
        "evaluation_seconds": seconds, "timing_scope": "forecast plus metric evaluation; not paper throughput",
        "selection": "minimum validation loss, earliest tie" if args.run else "parameter-free control"}
    summary.update(dataset_id=config["problem"] + "-" + args.split,
        bootstrap_population_id=config["problem"] + ("-weak" if args.split in ("weak", "veryweak") else "-" + args.split),
        fit_id=f"{config['problem']}-{method}-s{seed}" if args.run else config["problem"] + "-" + method,
        view="published_dispatch" if args.frame == "default" else "frame_" + frame,
        frame_velocity=[0., 0.] if config["problem"] == "swe2d" else [0.])
    write_json(args.output / "metrics.json", summary)
    if args.run is not None and args.split not in ("train", "validation"):
        write_json(args.run / "test-evaluated.json", {"checkpoint_sha256": sha256(args.run / "best.pt"), "training_closed": True})
    print("EVALUATED", config["problem"], method, frame, summary["endpoint_function_mean"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "validation", "weak", "veryweak"), default="test")
    parser.add_argument("--method", choices=METHODS, default="carrier_only")
    parser.add_argument("--seed", type=int, default=1621)
    parser.add_argument("--frame", choices=("default", "moving", "physical"), default="default")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
