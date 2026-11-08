"""Construct the paper architectures, including pre-optimizer branch folding."""
from pathlib import Path
import torch
from torch import nn
from .fusion import fuse_model
from .variants import make_variant

METHODS = ("dcino", "carrier_only", "fno", "unet", "fno_physical_wrapper", "no_sv", "no_dyadic", "raw_carrier")


class CarrierOnly(nn.Module):
    """The same analytic SV carrier and radial correction, with no learned tensors."""
    def __init__(self, problem, dt=.02, alpha=.05, power=2, repair=True):
        super().__init__()
        self.system, self.dt = problem, dt
        self.alpha, self.power, self.repair = alpha, power, repair
        self.scale = 1.0
        if problem == "swe2d":
            from .two_dimensional.swe2d import carrier, safety
            from .two_dimensional.spectral_viscosity2d import radial_multiplier
            self._carrier, self._safety, self._multiplier = carrier, safety, radial_multiplier
        else:
            from .one_dimensional.carrier_math import filtered_carrier
            from .one_dimensional.models_core import safety
            self._carrier, self._safety = filtered_carrier, safety

    def forward(self, state):
        if self.system == "swe2d":
            return self._carrier(state, self.dt, multiplier=self._multiplier(state, self.dt, self.alpha, self.power))
        return self._carrier(state, self.system, self.dt, self.alpha, self.power)

    def step(self, state):
        candidate = self(state)
        if not self.repair:
            return candidate, torch.ones(len(state), dtype=state.dtype, device=state.device)
        return (self._safety(state, candidate) if self.system == "swe2d"
                else self._safety(state, candidate, self.system))


def build_model(problem, method="dcino", seed=1621, dt=.02, initialization=None):
    """Build exact production widths; optionally restore a zero-update payload.

    For D-CINO, historical three-branch tensors are folded before creating the
    optimizer. A supplied fused state is loaded after that construction. No old
    optimizer state is transferred. Without an asset, tensors use this installed
    PyTorch version's CPU RNG; cross-version historical bit equality is not assumed.
    """
    if problem not in ("swe", "euler", "swe2d") or method not in METHODS:
        raise ValueError("Unknown problem or method")
    if method in ("carrier_only", "raw_carrier"):
        if initialization is not None:
            raise ValueError("A parameter-free carrier has no initialization checkpoint")
        return CarrierOnly(problem, dt=dt, repair=method != "raw_carrier")
    torch.manual_seed(seed)
    payload = None if initialization is None else torch.load(Path(initialization), map_location="cpu", weights_only=False)
    if payload is not None and (payload.get("optimizer_updates", payload.get("update", 0)) != 0):
        raise ValueError("Training initialization must be a zero-update state")
    if payload is not None and payload.get("seed", seed) != seed:
        raise ValueError("Initialization seed mismatch")
    if method in ("dcino", "no_sv", "no_dyadic"):
        if problem == "swe2d":
            from .two_dimensional.architectures import CarrierSV
            source = CarrierSV(width=24, dt=dt, alpha=.05, power=2)
            source.scale = 1.0
        else:
            from .one_dimensional.carrier_model import CarrierScratchMap
            source = CarrierScratchMap(problem, dt=dt, width=48, alpha=.05, power=2)
        historical = payload is not None and any(name.startswith("pointwise.") for name in payload["model"])
        if historical:
            source.load_state_dict(payload["model"], strict=True)
        model = fuse_model(source)[0] if method == "dcino" else make_variant(source, method)
        if payload is not None and not historical:
            model.load_state_dict(payload["model"], strict=True)
    elif problem == "swe2d":
        from .two_dimensional.baselines2d import build_model as baseline
        model = baseline({"fno": "fno_native", "unet": "unet_native"}.get(method, method), dt=dt)
        model.scale = 1.0
    else:
        from .one_dimensional.baselines import initialize
        model = initialize(method, problem, seed, dt=dt)
    if payload is not None and method not in ("dcino", "no_sv", "no_dyadic"):
        model.load_state_dict(payload["model"], strict=True)
    if payload is not None and "rng" in payload:
        from ..training_math import restore_rng
        restore_rng(payload["rng"])
    return model
