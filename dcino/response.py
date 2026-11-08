"""Fixed conservative weak-direction response with exact parameter promotion."""
import numpy as np
import torch

@torch.no_grad()
def response(model, physical, problem, method):
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    # This worker owns a freshly loaded checkpoint copy. In-place precision
    # promotion also supports the historical wrapper's Python module handle,
    # which cannot be deep-copied. No checkpoint artifact is changed.
    promoted = model._apply(lambda value: value.to(torch.complex128) if value.is_complex()
                           else value.to(torch.float64) if value.is_floating_point() else value)
    checks = {key: bool(torch.equal(value, promoted.state_dict()[key].to(value.dtype))) for key, value in before.items()}
    if not all(checks.values()):
        raise RuntimeError("Precision promotion altered checkpoint values")
    physical = physical.double()
    if problem == "swe2d":
        from .models.two_dimensional import swe2d as core
        from .models.two_dimensional.spectral_viscosity2d import radial_multiplier
        canonical = core.galilean(physical, -core.mean_velocity(physical))
        dimensions = (-2, -1)
        mean = canonical.mean(dimensions, keepdim=True).expand_as(canonical)
        direction = (canonical - mean) / .0625
        damping = radial_multiplier(mean, dt=.02, alpha=.05, power=2)
        bnu = core.carrier(mean + direction, .02, multiplier=damping) - mean
        undamped = core.carrier(mean + direction, .02) - mean
        scale = 1.
    else:
        from .models.one_dimensional import models_core as core
        from .models.one_dimensional.carrier_math import filtered_carrier
        velocity = physical[:, 1].sum(-1) / physical[:, 0].sum(-1)
        canonical = core.galilean(physical, -velocity, problem)
        dimensions = (-1,)
        mean = canonical.mean(dimensions, keepdim=True).expand_as(canonical)
        direction = (canonical - mean) / .0625
        bnu = filtered_carrier(mean + direction, problem, .02, .05, 2) - mean
        undamped = core.characteristic_propagate(mean + direction, problem, .02) - mean
        scale = torch.tensor([1., 1., 2.5] if problem == "euler" else [1., 1.], device=physical.device, dtype=torch.float64).reshape(1, -1, 1)

    def rms(value):
        return float((value / scale).square().mean().sqrt())

    def candidate(state):
        raw = promoted(state)
        return state + promoted.dt * (raw - raw.mean(dimensions, keepdim=True)) if method == "fno_physical_wrapper" else raw

    def step(state):
        if method == "raw_carrier":
            return promoted(state), torch.ones(len(state), device=state.device, dtype=state.dtype)
        return promoted.step(state)

    full0, theta0 = step(mean)
    pre0 = candidate(mean)
    values = []
    for epsilon in np.logspace(-6, -1, 11):
        full, theta = step(mean + epsilon * direction)
        pre = candidate(mean + epsilon * direction)
        values.append({"epsilon": float(epsilon), "full_step_Bnu_remainder_rms": rms(full - full0 - epsilon * bnu),
            "preQ_Bnu_remainder_rms": rms(pre - pre0 - epsilon * bnu),
            "full_step_undamped_B_remainder_rms": rms(full - full0 - epsilon * undamped),
            "preQ_undamped_B_remainder_rms": rms(pre - pre0 - epsilon * undamped),
            "full_step_physical_constant_error_rms": rms(full - mean - epsilon * bnu), "theta": float(theta.mean()),
            "full_finite": bool(torch.isfinite(full).all()), "preQ_finite": bool(torch.isfinite(pre).all())})
    energy = torch.fft.rfftn(direction / scale, dim=dimensions).abs().square().mean((0, 1))
    weights = torch.full_like(energy, 2.)
    weights[..., 0] = weights[..., -1] = 1.
    energy = energy * weights
    energy /= energy.sum()
    if problem == "swe2d":
        kx = torch.fft.rfftfreq(physical.shape[-1], d=1/physical.shape[-1], device=physical.device)
        ky = torch.fft.fftfreq(physical.shape[-2], d=1/physical.shape[-2], device=physical.device)
        high = (kx[None, :].square() + ky[:, None].square()).sqrt() > int(np.ceil(np.sqrt(physical.shape[-1])))
        fraction = float(energy[high].sum())
    else:
        fraction = float(energy[int(np.ceil(np.sqrt(physical.shape[-1])))+1:].sum())
    arrays = {"physical_initial": physical.cpu().numpy(), "canonical_initial": canonical.cpu().numpy(),
              "mean": mean.cpu().numpy(), "direction": direction.cpu().numpy(),
              "Bnu_direction": bnu.cpu().numpy(), "undamped_B_direction": undamped.cpu().numpy(),
              "direction_rfft_energy_fractions": energy.cpu().numpy()}
    result = {"values": values, "full_step_constant_bias_rms": rms(full0 - mean),
              "preQ_constant_bias_rms": rms(pre0 - mean), "constant_theta": float(theta0.mean()),
              "direction_rms": rms(direction), "carrier_Bnu_minus_undamped_direction_rms": rms(bnu - undamped),
              "direction_energy_above_unchanged_cutoff": fraction, "cutoff": int(np.ceil(np.sqrt(physical.shape[-1]))),
              "precision_conversion": {"roundtrip_all_tensors_equal": True, "tensor_checks": checks}}
    return result, arrays


def main():
    import argparse
    from pathlib import Path
    from .io import configure, load_split, read_json, write_json
    from .models.factory import build_model, METHODS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Directory containing weak.npz or a weak entry in data-map.json")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--method", choices=METHODS, default="carrier_only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    configure(args.threads)
    config = read_json(args.config)
    method = args.method
    if args.run:
        if read_json(args.run / "status.json")["state"] != "completed":
            raise ValueError("Response requires a completed fit")
        payload = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
        method = payload["config"]["method"]
        if payload["config"]["seed"] != 1621:
            raise ValueError("The prescribed response diagnostic uses seed 1621")
        for key in ("problem", "cells", "dt", "profile"):
            if payload["config"][key] != config[key]:
                raise ValueError("Response configuration differs from the fit")
    elif method not in ("carrier_only", "raw_carrier"):
        raise ValueError("A learned response requires --run")
    model = build_model(config["problem"], method)
    if args.run:
        model.load_state_dict(payload["model"], strict=True)
    model = model.to(args.device).eval()
    sequence, labels, ids, binding = load_split(args.data, "weak", config, dtype=np.float64)
    index = labels.tolist().index("smooth")
    physical = sequence[index:index + 1, 0].to(args.device)
    probe, arrays = response(model, physical, config["problem"], method)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "direction.npz", **arrays)
    write_json(args.output / "response.json", {"system": config["problem"], "method": method,
        "profile": config["profile"], "function_id": ids[index], "data": binding, **probe})
    print("RESPONSE", config["problem"], method, probe["precision_conversion"]["roundtrip_all_tensors_equal"], flush=True)


if __name__ == "__main__":
    main()
