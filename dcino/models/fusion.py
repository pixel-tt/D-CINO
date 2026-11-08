"""Absorb each D-CINO pointwise affine map into its local stencil.

For a five-tap/circular 5x5 stencil, L(z) + P(z) has weights
K_fused[center] = K[center] + A and bias b_fused = b + a.  This
preserves the real-arithmetic function class. Floating-point association and
rounding change, so output/repair/rollout equality must be measured separately.
Historical carrier, viscosity, mean projection, repair and frame code are reused.
"""
from __future__ import annotations

import copy
import hashlib
import importlib

import torch
from torch import nn
from torch.nn import functional


def tensor_sha256(state):
    """Hash tensor names, NumPy dtypes, shapes and bytes as in the parent bundle."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update((name + str(array.dtype) + str(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def real_parameter_count(model):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())


def cast_model(model, dtype, device=None):
    """Copy a model with matching real/complex precision, including FFT weights.

    Use this rather than Module.double(), which leaves complex64 parameters
    unchanged in PyTorch. Integral/bool buffers retain their dtype.
    """
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Expected float32 or float64")
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    result = copy.deepcopy(model)

    def convert(tensor):
        target = complex_dtype if tensor.is_complex() else dtype if tensor.is_floating_point() else tensor.dtype
        return tensor.to(device=device, dtype=target)

    return result._apply(convert)


def _merge_pointwise(model, dimension):
    """Perform the central-tap and bias additions, then remove the branch."""
    if len(model.pointwise) != len(model.local) or len(model.spectral) != len(model.local):
        raise ValueError("Spectral, pointwise and local depths must agree")
    records = []
    for index, (pointwise, local) in enumerate(zip(model.pointwise, model.local)):
        width = local.weight.shape[0]
        if tuple(pointwise.weight.shape) != (width, width) + (1,) * dimension:
            raise ValueError("Expected an unconstrained width-by-width 1x1 pointwise map")
        if tuple(local.weight.shape) != (width, width) + (5,) * dimension:
            raise ValueError("Expected an unconstrained five-tap local stencil")
        if pointwise.bias is None or local.bias is None:
            raise ValueError("Both historical affine branches must have biases")
        if pointwise.groups != 1 or pointwise.stride != (1,) * dimension:
            raise ValueError("Pointwise map must preserve each spatial location")
        if pointwise.padding != (0,) * dimension or pointwise.dilation != (1,) * dimension:
            raise ValueError("Pointwise map must have zero padding and unit dilation")
        if dimension == 2 and (local.padding_mode != "circular" or local.padding != (2, 2)
                               or local.stride != (1, 1) or local.dilation != (1, 1) or local.groups != 1):
            raise ValueError("Expected the historical circular 5x5 convolution")
        if local.weight.dtype != pointwise.weight.dtype or local.weight.device != pointwise.weight.device:
            raise ValueError("Branch dtype/device mismatch")
        if local.weight.requires_grad != pointwise.weight.requires_grad or local.bias.requires_grad != pointwise.bias.requires_grad:
            raise ValueError("Branches must have matching trainability before reparameterization")
        center = (slice(None), slice(None)) + (2,) * dimension
        matrix = pointwise.weight.reshape(width, width)
        exact_weight = local.weight[center].detach().double() + matrix.detach().double()
        exact_bias = local.bias.detach().double() + pointwise.bias.detach().double()
        with torch.no_grad():
            local.weight[center].add_(matrix)
            local.bias.add_(pointwise.bias)
        records.append({
            "block": index, "center": [2] * dimension,
            "weight_addition_roundoff_linf": float((local.weight[center].detach().double() - exact_weight).abs().max()),
            "bias_addition_roundoff_linf": float((local.bias.detach().double() - exact_bias).abs().max()),
        })
    del model.pointwise
    return records


class FusedCarrier1D(nn.Module):
    """Historical CarrierScratchMap with only spectral + local in each block."""

    def __init__(self, source):
        super().__init__()
        if type(source).__name__ != "CarrierScratchMap":
            raise TypeError("Expected the historical 1D CarrierScratchMap")
        historical = importlib.import_module(type(source).__module__)
        copied = copy.deepcopy(source)
        self.__dict__.update(copied.__dict__)
        self._filtered_carrier = historical.historical.filtered_carrier
        self._safety = historical.original.safety
        self.fusion_blocks = _merge_pointwise(self, dimension=1)

    def forward(self, state):
        mean = state.mean(-1, keepdim=True)
        features = torch.cat((state / self.scale, (state + mean) / (2 * self.scale)), 0)
        latent = self.lift(features)
        for spectral, local in zip(self.spectral, self.local):
            latent = functional.gelu(spectral(latent) + local(latent))
        full, half = self.output(latent).chunk(2, dim=0)
        increment = .2 * self.scale * (2 * (full - 2 * half))
        carrier = self._filtered_carrier(state, self.system, self.dt, self.alpha, self.power)
        return carrier + increment - increment.mean(-1, keepdim=True)

    def step(self, state):
        return self._safety(state, self(state), self.system)


class FusedCarrier2D(nn.Module):
    """Historical CarrierSV with only spectral + local in each block."""

    def __init__(self, source):
        super().__init__()
        if type(source).__name__ != "CarrierSV":
            raise TypeError("Expected the historical 2D CarrierSV")
        historical = importlib.import_module(type(source).__module__)
        copied = copy.deepcopy(source)
        self.__dict__.update(copied.__dict__)
        self._carrier = historical.carrier
        self._radial_multiplier = historical.radial_multiplier
        self._safety = historical.safety
        self.fusion_blocks = _merge_pointwise(self, dimension=2)

    def forward(self, state):
        mean = state.mean((-2, -1), keepdim=True)
        latent = self.lift(torch.cat((state, (state + mean) / 2), 0))
        for spectral, local in zip(self.spectral, self.local):
            latent = functional.gelu(spectral(latent) + local(latent))
        full, half = self.output(latent).chunk(2, 0)
        increment = .2 * (2 * (full - 2 * half))
        damping = self._radial_multiplier(state, self.dt, self.alpha, self.power)
        return self._carrier(state, self.dt, multiplier=damping) + increment - increment.mean((-2, -1), keepdim=True)

    def step(self, state):
        return self._safety(state, self(state))


def fuse_model(model):
    """Return (independent fused model, provenance), preserving dtype/device/mode.

    Accepts a selected checkpoint model or a freshly initialized historical
    model. For retraining, create the optimizer AFTER this conversion; optimizer
    moments and weight-decay dynamics are not invariant under reparameterization.
    Save/load the fused state_dict using a newly fused matching constructor.
    """
    original_hash = tensor_sha256(model.state_dict())
    dimension = model.local[0].weight.ndim - 2
    if dimension == 1:
        result = FusedCarrier1D(model)
    elif dimension == 2:
        result = FusedCarrier2D(model)
    else:
        raise ValueError("Only historical 1D/2D D-CINO models are supported")
    provenance = {
        "source_tensor_sha256": original_hash,
        "fused_tensor_sha256": tensor_sha256(result.state_dict()),
        "original_real_parameters": real_parameter_count(model),
        "fused_real_parameters": real_parameter_count(result),
        "removed_real_parameters": real_parameter_count(model) - real_parameter_count(result),
        "blocks": result.fusion_blocks,
        "arithmetic_claim": "Pointwise real-arithmetic equality; floating-point bitwise equality is not guaranteed",
        "optimizer_state_transferred": False,
    }
    if tensor_sha256(model.state_dict()) != original_hash:
        raise RuntimeError("Source model was mutated")
    if any(name.startswith("pointwise.") for name in result.state_dict()) or hasattr(result, "pointwise"):
        raise RuntimeError("Independent pointwise branch remains")
    return result, provenance
