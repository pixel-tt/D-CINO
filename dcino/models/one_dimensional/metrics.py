"""Exact historical native forecast and scoring policy; local dependencies."""
import time
import numpy as np
import torch
NUMERICAL_CAP=1e6
PHYSICAL_FLOOR=1e-7
DENOMINATOR_FLOOR=1e-14

def safe_json(value):
    """Undefined physical diagnostics become null, never primary score omissions."""
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def scale_array(system):
    return np.asarray([1., 1.] if system == "swe" else [1., 1., 2.5])


def physical_diagnostics(states, system):
    """Conservative layout (...,components,cells), with undefined entropy masked.

    A zero or negative density can make pressure undefined even if conservative
    state is finite. This affects validity/entropy diagnostics alone.
    """
    density = states[..., 0, :]
    with np.errstate(all="ignore"):
        pressure = .4*(states[..., 2, :]-.5*states[..., 1, :]**2/density) if system == "euler" else None
        valid = np.isfinite(states).all(axis=(-2, -1)) & (density >= PHYSICAL_FLOOR).all(axis=-1)
        if system == "euler":
            valid &= np.isfinite(pressure).all(axis=-1) & (pressure >= PHYSICAL_FLOOR).all(axis=-1)
            entropy_cells = -density*(np.log(pressure)-1.4*np.log(density))/.4
        else:
            entropy_cells = .5*states[..., 1, :]**2/density+.5*density**2
        entropy = np.where(valid, entropy_cells.mean(axis=-1), np.nan)
    return {"physical_invalid": ~valid, "min_density_or_depth": density.min(axis=-1),
            "min_pressure": pressure.min(axis=-1) if pressure is not None else None,
            "entropy": entropy}


def numerical_bad(state, scale):
    return not bool(torch.isfinite(state).all()) or bool((state/scale).abs().amax() > NUMERICAL_CAP)


def candidate_and_step(model, state, method, original):
    """Expose the actual pre-Q candidate while applying each unchanged map once."""
    raw = model(state)
    if method == "fno_physical_wrapper":
        candidate = state+model.dt*(raw-raw.mean(-1, keepdim=True))
        next_state, theta = model.original.safety(state, candidate, model.system)
    elif method.startswith("d_cino_"):
        candidate = raw
        next_state, theta = original.safety(state, candidate, model.system)
    else:
        candidate, next_state = raw, raw
        theta = torch.ones(state.shape[0], dtype=state.dtype, device=state.device)
    return candidate, next_state, theta, raw.mean(-1)


@torch.inference_mode()
def forecast(model, initial, steps, method, original):
    """Retain actual native/canonical and rendered states, including invalid ones.

    Frame/Q-bearing models follow the inherited equations exactly. Unlike the
    earlier evaluator, this accuracy policy records finite physical invalidity
    without making it a numerical failure. Observed states never feed back into
    the canonical trajectory. Independent functions are advanced separately,
    matching frozen D-CINO's existing inference implementation.
    """
    wrapped = method == "fno_physical_wrapper" or method.startswith("d_cino_")
    observations, canonicals, factors, masks, candidates, raw_means, failures = [], [], [], [], [], [], []
    timings = []
    for index, field in enumerate(initial.split(1)):
        started = time.perf_counter()
        velocity = field[:, 1].sum(-1)/field[:, 0].sum(-1) if wrapped else None
        current = original.galilean(field, -velocity, model.system) if wrapped else field
        output, canonical, theta_history = [field], [current], []
        candidate_means, neural_means = [], []
        failed = numerical_bad(current, model.scale) or numerical_bad(field, model.scale)
        failure_history = [failed]
        if failed:
            failures.append({"function_index": index, "first_step": 0, "reason": "Invalid numerical initial state"})
        for step in range(1, steps+1):
            candidate_mean = torch.full_like(current.mean(-1), float("nan"))
            raw_mean = candidate_mean
            if not failed:
                try:
                    candidate, advanced, theta, raw_mean = candidate_and_step(model, current, method, original)
                    candidate_mean = candidate.mean(-1)
                    rendered = (original.galilean(original.translate(advanced,
                                velocity*(step*model.dt*field.shape[-1])), velocity, model.system)
                                if wrapped else advanced)
                    if numerical_bad(advanced, model.scale) or numerical_bad(rendered, model.scale):
                        raise FloatingPointError("Nonfinite or max_abs(U/S)>1e6 actual state")
                    current = advanced
                except Exception as exception:
                    failed = True
                    failures.append({"function_index": index, "first_step": step,
                                     "reason": f"{type(exception).__name__}: {exception}"})
            if failed:
                rendered = output[-1]
                theta = torch.zeros(1, dtype=field.dtype, device=field.device)
            output.append(rendered)
            canonical.append(current)
            theta_history.append(theta)
            candidate_means.append(candidate_mean)
            neural_means.append(raw_mean)
            failure_history.append(failed)
        observations.append(torch.cat(output, dim=0).cpu().numpy())
        canonicals.append(torch.cat(canonical, dim=0).cpu().numpy())
        factors.append(torch.cat(theta_history).cpu().numpy())
        masks.append(failure_history)
        candidates.append(torch.cat(candidate_means).cpu().numpy())
        raw_means.append(torch.cat(neural_means).cpu().numpy())
        timings.append(time.perf_counter()-started)
    return {"prediction_native": np.stack(observations), "canonical_native": np.stack(canonicals),
            "theta": np.stack(factors), "numerical_failure_native": np.asarray(masks, dtype=bool),
            "pre_Q_candidate_mean": np.stack(candidates), "raw_network_mean": np.stack(raw_means),
            "failures": failures, "seconds_per_function": timings,
            "frame": "canonical trajectory with deferred physical observations" if wrapped else "native physical autoregression"}


def metrics(prediction, reference, system, numerical_failure):
    """Physical error retains mean bias; validity never censors finite accuracy."""
    scale = scale_array(system)[None, None, :, None]
    with np.errstate(all="ignore"):
        error = prediction.astype(np.float64)-reference
        state_rms = np.sqrt(np.mean((error/scale)**2, axis=(-2, -1)))
        reference_fluctuation = reference-reference.mean(-1, keepdims=True)
        endpoint_amplitude = np.sqrt(np.mean((reference_fluctuation/scale)**2, axis=(-2, -1)))
        initial_amplitude = endpoint_amplitude[:, :1]
        fluctuation_error = (prediction-prediction.mean(-1, keepdims=True))-reference_fluctuation
        fluctuation_rms = np.sqrt(np.mean((fluctuation_error/scale)**2, axis=(-2, -1)))
        result = {
            "S_scaled_state_RMS": state_rms,
            "weak_physical_RMS_over_endpoint_truth_fluctuation": state_rms/np.maximum(endpoint_amplitude, DENOMINATOR_FLOOR),
            "weak_physical_RMS_over_initial_fluctuation": state_rms/np.maximum(initial_amplitude, DENOMINATOR_FLOOR),
            "centered_fluctuation_RMS_over_endpoint_truth_fluctuation": fluctuation_rms/np.maximum(endpoint_amplitude, DENOMINATOR_FLOOR),
            "conservative_component_L1": np.abs(error).mean(-1),
            "conservative_component_L2": np.sqrt(np.mean(error**2, axis=-1)),
            "S_scaled_mean_error_RMS": np.sqrt(np.mean((error.mean(-1)/scale_array(system))**2, axis=-1))}
    for key, value in result.items():
        mask = numerical_failure[:, :, None] if value.ndim == 3 else numerical_failure
        result[key] = np.where(mask | ~np.isfinite(value), NUMERICAL_CAP, value)
    return result, {"endpoint_truth_fluctuation_RMS": endpoint_amplitude,
                    "initial_truth_fluctuation_RMS": initial_amplitude[:, 0],
                    "endpoint_denominator_floor_applied": endpoint_amplitude < DENOMINATOR_FLOOR,
                    "initial_denominator_floor_applied": initial_amplitude[:, 0] < DENOMINATOR_FLOOR}
