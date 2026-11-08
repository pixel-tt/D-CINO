"""Exact original wide24 carrier-SV and published baseline architectures."""
import torch
from .models2d import FlowMap
from .spectral_viscosity2d import radial_multiplier
from .swe2d import carrier, safety


class CarrierSV(FlowMap):
    """Shared dyadic architecture with scalar radial damping on the carrier only."""
    def __init__(self, width=12, dt=.02, alpha=.05, power=2):
        super().__init__('dyadic', width=width, modes=4, depth=3, dt=dt)
        self.alpha, self.power = alpha, power

    def forward(self, u):
        mean = u.mean((-2, -1), keepdim=True)
        z = self.lift(torch.cat((u, (u+mean)/2), 0))
        for spectral, pointwise, local in zip(self.spectral, self.pointwise, self.local):
            z = torch.nn.functional.gelu(spectral(z)+pointwise(z)+local(z))
        full, half = self.output(z).chunk(2, 0)
        increment = .2*(2*(full-2*half))
        damping = radial_multiplier(u, self.dt, self.alpha, self.power)
        return carrier(u, self.dt, multiplier=damping)+increment-increment.mean((-2, -1), keepdim=True)

    def step(self, u):
        return safety(u, self(u))
