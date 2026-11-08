"""Two single-component changes to the current folded D-CINO architecture.

The analytic carrier, fixed input/channel scales, kappa=.2, mean projection,
radial correction and outer forecast are inherited. No-SV sets alpha=0. No-dyadic
replaces 2*(Phi(U)-2*Phi((U+mean(U))/2)) by Phi(U), a single network call.
"""
from __future__ import annotations

import torch
from torch.nn import functional

from .fusion import FusedCarrier1D, FusedCarrier2D


class AblationCarrier1D(FusedCarrier1D):
    def __init__(self, source, variant):
        super().__init__(source)
        if variant not in ("no_sv", "no_dyadic"):
            raise ValueError("Unknown variant")
        self.variant = variant
        if variant == "no_sv":
            self.alpha = 0.0

    def forward(self, state):
        if self.variant == "no_sv":
            return super().forward(state)
        latent = self.lift(state / self.scale)
        for spectral, local in zip(self.spectral, self.local):
            latent = functional.gelu(spectral(latent) + local(latent))
        increment = .2 * self.scale * self.output(latent)
        carrier = self._filtered_carrier(state, self.system, self.dt, self.alpha, self.power)
        return carrier + increment - increment.mean(-1, keepdim=True)


class AblationCarrier2D(FusedCarrier2D):
    def __init__(self, source, variant):
        super().__init__(source)
        if variant not in ("default", "no_sv", "no_dyadic"):
            raise ValueError("Unknown variant")
        self.variant = variant
        if variant == "no_sv":
            self.alpha = 0.0

    def forward(self, state):
        if self.variant != "no_dyadic":
            return super().forward(state)
        latent = self.lift(state)
        for spectral, local in zip(self.spectral, self.local):
            latent = functional.gelu(spectral(latent) + local(latent))
        increment = .2 * self.output(latent)
        damping = self._radial_multiplier(state, self.dt, self.alpha, self.power)
        return self._carrier(state, self.dt, multiplier=damping) + increment - increment.mean((-2, -1), keepdim=True)


def make_variant(source, variant):
    """Fold a fresh historical zero-update model, then change one operation."""
    return (AblationCarrier2D(source, variant) if source.local[0].weight.ndim == 4
            else AblationCarrier1D(source, variant))
