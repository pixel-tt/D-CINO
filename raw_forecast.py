"""Current unrepaired raw-carrier diagnostic; preserves finite invalidity."""
import numpy as np
import torch

@torch.no_grad()
def raw_forecast_1d(model, initial, steps, frame):
    """Raw-carrier diagnostic retains finite physical invalidity without Q."""
    from .models_core import galilean, translate
    predictions, masks, factors, issues = [], [], [], []
    for index, field in enumerate(initial.split(1)):
        velocity = field[:, 1].sum(-1) / field[:, 0].sum(-1)
        current = galilean(field, -velocity, model.system) if frame == "moving" else field
        out, failure, theta = [field], [False], []
        failed = False
        for step in range(1, steps + 1):
            if not failed:
                try:
                    candidate = model(current)
                    rendered = (galilean(translate(candidate, velocity * (step * model.dt * field.shape[-1])), velocity, model.system)
                                if frame == "moving" else candidate)
                    if not bool(torch.isfinite(candidate).all() & torch.isfinite(rendered).all()):
                        raise FloatingPointError("Nonfinite raw carrier state")
                    current = candidate
                except Exception as error:
                    failed = True
                    issues.append({"function_index": index, "first_step": step, "reason": str(error)})
            if failed:
                rendered = out[-1]
            out.append(rendered)
            failure.append(failed)
            theta.append(torch.full((1,), 0. if failed else 1., device=field.device, dtype=field.dtype))
        predictions.append(torch.stack(out, 1))
        masks.append(failure)
        factors.append(torch.stack(theta, 1))
    prediction = torch.cat(predictions).cpu().numpy()
    from .metrics import physical_diagnostics
    return {"prediction_native": prediction, "theta": torch.cat(factors).cpu().numpy(),
            "numerical_failure_native": np.asarray(masks),
            "physical_invalid_native": physical_diagnostics(prediction, model.system)["physical_invalid"],
            "failures": issues}
