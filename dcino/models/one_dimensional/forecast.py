"""Exact original canonical/deferred D-CINO inference; local import wiring."""
import time
import torch
from .models_core import primitive, galilean, translate

def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def admissible(u, system, floor=1e-7):
    if not bool(torch.isfinite(u).all()) or not bool((u[:, 0] >= floor).all()):
        return False
    return system == "swe" or bool((primitive(u, system)[:, 2] >= floor).all())


@torch.no_grad()
def rollout(model, initial, steps, dt=.02, frame="moving"):
    """Free co-moving rollout using actual filtered steps and observation remaps.

    Each function advances separately so an exception affects only that function.
    A failure is absorbing; its last valid observation is a storage placeholder,
    always accompanied by a failure mask. Callers must use the mask and penalty.
    Rendered observations are never fed back into the canonical trajectory.
    Returns observations, repair factors, failure masks and timing/failure detail.
    """
    if frame not in ("moving", "physical"):
        raise ValueError("frame must be moving or physical")
    if steps < 1 or initial.ndim != 3:
        raise ValueError("Positive steps and (batch, components, cells) required")
    observations, retained, failure_masks, times, failures = [], [], [], [], []
    for index, u in enumerate(initial.split(1)):
        synchronize(u.device)
        start = time.perf_counter()
        failed = not admissible(u, model.system)
        velocity = u[:, 1].sum(-1) / u[:, 0].sum(-1) if frame == "moving" else None
        current = galilean(u, -velocity, model.system) if frame == "moving" else u
        out, factors, mask = [u], [], [failed]
        if failed:
            failures.append({"index": index, "first_step": 0, "reason": "Invalid initial state"})
        for k in range(1, steps+1):
            if not failed:
                try:
                    candidate, theta = model.step(current)
                    if not admissible(candidate, model.system):
                        raise FloatingPointError("Inadmissible canonical state")
                    rendered = (galilean(translate(candidate, velocity * (k*dt*u.shape[-1])),
                                         velocity, model.system) if frame == "moving" else candidate)
                    if not admissible(rendered, model.system):
                        raise FloatingPointError("Inadmissible observed state")
                    current = candidate
                except Exception as exc:
                    failed = True
                    failures.append({"index": index, "first_step": k,
                                     "reason": f"{type(exc).__name__}: {exc}"})
            if failed:
                rendered = out[-1]
                theta = torch.zeros(1, dtype=u.dtype, device=u.device)
            out.append(rendered)
            factors.append(theta)
            mask.append(failed)
        synchronize(u.device)
        times.append(time.perf_counter()-start)
        observations.append(torch.stack(out, 1))
        retained.append(torch.stack(factors, 1))
        failure_masks.append(mask)
    return (torch.cat(observations), torch.cat(retained),
            torch.tensor(failure_masks, dtype=torch.bool, device=initial.device),
            {"seconds_per_function": times, "failures": failures,
             "remapping": "linear, observation only; canonical state retained" if frame == "moving" else "none; physical state retained"})


from .metrics import metrics, physical_diagnostics, safe_json, scale_array


def forecast(model, initial, steps, method=None, original=None, frame="moving"):
    """Adapt the actual D-CINO primary rollout and preserve its strict mask.

    The inherited evaluator calls dcino_rollout for D-CINO, while native
    baselines use evaluation_math.forecast. Finite physical invalidity
    is an absorbing D-CINO failure in BOTH frame arms, as historically run.
    method/original are accepted for caller compatibility only.
    """
    values, theta, masks, detail = rollout(model, initial, steps, model.dt, frame=frame)
    predictions = values.cpu().numpy()
    return {"prediction_native": predictions, "theta": theta.cpu().numpy(),
            "numerical_failure_native": masks.cpu().numpy(),
            "inherited_failure_native": masks.cpu().numpy(),
            "physical_invalid_native": physical_diagnostics(predictions, model.system)["physical_invalid"],
            "failures": detail["failures"], "seconds_per_function": detail["seconds_per_function"],
            "frame": frame, "remapping": detail["remapping"]}
