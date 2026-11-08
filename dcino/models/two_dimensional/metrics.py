"""Byte-faithful function copies from frozen WEAK-BASELINES-2D evaluator."""
import numpy as np
import torch
from .swe2d import entropy,galilean,mean_velocity,translate
PENALTY=1e6
CATASTROPHIC_STATE_LINF=1e6


@torch.no_grad()
def predict(model, initial, steps, wrapped):
    """Store failed raw candidates and use a transparent absorbing failure mask.

    Finite negative-depth outputs continue and retain their actual errors.
    Only nonfinite or max-absolute-state greater than 1e6 is an absorbing
    numerical failure. Its raw candidate is stored and affected endpoints
    receive the same frozen failure cap in every method.
    """
    predictions, internals, thetas, masks, failures = [], [], [], [], []
    for index, physical in enumerate(initial.split(1)):
        velocity = mean_velocity(physical)
        current = galilean(physical, -velocity) if wrapped else physical
        observed, states, qs, failed_mask = [physical], [current], [], [False]
        failed = False
        for step in range(1, steps + 1):
            if not failed:
                current, theta = model.step(current)
                rendered = (galilean(translate(current, velocity * (step * model.dt * physical.shape[-1])), velocity)
                            if wrapped else current)
                if not bool(torch.isfinite(current).all() & torch.isfinite(rendered).all()
                            & (current.abs().amax() <= CATASTROPHIC_STATE_LINF)
                            & (rendered.abs().amax() <= CATASTROPHIC_STATE_LINF)):
                    failed = True
                    failures.append(dict(index=index, first_step=step,
                                         reason='nonfinite or maxabsstate > 1e6; raw candidate retained'))
            else:
                theta = torch.zeros(1, dtype=initial.dtype, device=initial.device)
            observed.append(rendered)
            states.append(current)
            qs.append(theta)
            failed_mask.append(failed)
        predictions.append(torch.stack(observed, dim=1))
        internals.append(torch.stack(states, dim=1))
        thetas.append(torch.stack(qs, dim=1))
        masks.append(failed_mask)
    return (torch.cat(predictions), torch.cat(internals), torch.cat(thetas),
            torch.tensor(masks, dtype=torch.bool, device=initial.device), failures)


def metrics(prediction, truth, failed, initial_reference=None):
    """Conservative physical-state errors normalized by the true fluctuation.

    fluctuation_l2 includes constant-offset errors in the numerator. The
    centered_shape_l2 and normalized_mean_l2 diagnostics separate those errors
    without replacing the primary physical-output metric.
    """
    prediction, truth = prediction.double(), truth.double()
    invalid = (failed | ~torch.isfinite(prediction).all((2, 3, 4))
               | (prediction.abs().amax((2, 3, 4)) > CATASTROPHIC_STATE_LINF))
    invalid = invalid.cumsum(1) > 0
    safe = torch.where(invalid[:, :, None, None, None], truth, prediction)
    error = safe - truth
    ref_mean = truth.mean((-2, -1), keepdim=True)
    pred_mean = safe.mean((-2, -1), keepdim=True)
    reference_rms = (truth - ref_mean).square().mean((2, 3, 4)).sqrt()
    denominator = reference_rms.clamp_min(1e-14)
    if initial_reference is None:
        initial_denominator = denominator[:, :1]
        initial_rms = reference_rms[:, :1]
    else:
        initial_reference = initial_reference.double()
        initial_rms = (initial_reference - initial_reference.mean((-2, -1), keepdim=True)).square().mean((2, 3, 4)).sqrt()
        initial_denominator = initial_rms.clamp_min(1e-14)
    centered_error = (safe - pred_mean) - (truth - ref_mean)
    values = dict(
        state_rms=error.square().mean((2, 3, 4)).sqrt(),
        fluctuation_l2=error.square().mean((2, 3, 4)).sqrt() / denominator,
        initial_fluctuation_l2=error.square().mean((2, 3, 4)).sqrt() / initial_denominator,
        centered_shape_l2=centered_error.square().mean((2, 3, 4)).sqrt() / denominator,
        normalized_mean_l2=(pred_mean - ref_mean).square().mean((2, 3, 4)).sqrt() / denominator,
        component_l2=error.square().mean((-2, -1)).sqrt(),
        mean_linf=(pred_mean - ref_mean).abs().amax((2, 3, 4)),
        depth_error_tv=((error[:, :, 0] - error[:, :, 0].roll(1, -1)).abs()
                        + (error[:, :, 0] - error[:, :, 0].roll(1, -2)).abs()).sum((-2, -1)) / truth.shape[-1])
    result = {}
    for name, value in values.items():
        mask = invalid[..., None] if value.ndim == 3 else invalid
        result[name] = torch.where(mask, PENALTY, value).cpu().tolist()
    result['failure_mask'] = invalid.cpu().tolist()
    result['physical_invalid_mask'] = ((prediction[:, :, 0].amin((-2, -1)) < 1e-7)
                                       | ~torch.isfinite(prediction).all((2, 3, 4))).cpu().tolist()
    result['reference_fluctuation_rms'] = reference_rms.cpu().tolist()
    result['reference_floor_active_mask'] = (reference_rms < 1e-14).cpu().tolist()
    result['initial_reference_floor_active_mask'] = (initial_rms < 1e-14).expand_as(reference_rms).cpu().tolist()
    return result


def entropy_diagnostic(internal, truth, failed):
    """Return undefined/failed entropy as null with an explicit validity mask."""
    valid = (~failed & torch.isfinite(internal).all((2, 3, 4))
             & (internal[:, :, 0].amin((-2, -1)) >= 1e-7))
    safe = torch.where(valid[:, :, None, None, None], internal, truth.to(internal.dtype))
    values = entropy(safe.reshape(-1, *internal.shape[2:])).mean((-2, -1)).reshape(*internal.shape[:2])
    valid &= torch.isfinite(values)
    return dict(internal_entropy=[[float(value) if bool(ok) else None for value, ok in zip(row, flags)]
                                  for row, flags in zip(values.cpu(), valid.cpu())],
                entropy_defined_mask=valid.cpu().tolist())


def reduce_families(result, labels, endpoints):
    labels = np.asarray(labels)
    summaries = {}
    for family in ('all', 'smooth', 'dam'):
        indices = np.arange(len(labels)) if family == 'all' else np.flatnonzero(labels == family)
        if not len(indices):
            continue
        summary = dict(function_count=len(indices))
        for key, value in result.items():
            if key == 'reference_fluctuation_rms':
                continue
            values = np.asarray(value)[indices][:, endpoints]
            summary[key] = values.mean(axis=0).tolist()
            if key.endswith('floor_active_mask'):
                summary[key.replace('active_mask', 'activation_count')] = values.sum(axis=0).tolist()
        summaries[family] = summary
    return summaries
