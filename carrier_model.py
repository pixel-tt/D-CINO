"""Unchanged scratch D-CINO constructor/forward/repair; local imports."""
import torch
import torch.nn.functional as functional
from . import models_core as original
from . import carrier_math as historical

class CarrierScratchMap(original.FlowMap):
    """One native interval with independently random width24 or width48 weights.

    Parameters select the conservative system, native time increment, residual
    width and fixed spectral viscosity. No mixing or box projection is applied.
    The local stencil, 16 Fourier modes, depth3 and kappa=.2 are prescribed.
    """
    def __init__(self, system, dt=.02, width=24, alpha=.05, power=2):
        if system not in ("swe", "euler") or width not in (24, 48):
            raise ValueError("Expected SWE/Euler with width24/48")
        if dt not in (.02, .04, .08):
            raise ValueError("Expected dt=.02/.04/.08")
        super().__init__(system, width=width, modes=16, depth=3, local=True,
                         characteristic=True, quadratic=False, normalized=False,
                         dyadic=True, dt=dt)
        self.width, self.alpha, self.power = width, alpha, power

    def forward(self, state):
        mean = state.mean(-1, keepdim=True)
        features = torch.cat((state/self.scale, (state+mean)/(2*self.scale)), 0)
        latent = self.lift(features)
        for index, (spectral, pointwise) in enumerate(zip(self.spectral, self.pointwise)):
            latent = functional.gelu(spectral(latent)+pointwise(latent)+self.local[index](latent))
        full, half = self.output(latent).chunk(2, dim=0)
        increment = .2*self.scale*(2*(full-2*half))
        carrier = historical.filtered_carrier(state, self.system, self.dt, self.alpha, self.power)
        return carrier+increment-increment.mean(-1, keepdim=True)

    def step(self, state):
        """Apply inherited positivity/total-entropy radial correction after the map."""
        return original.safety(state, self(state), self.system)


def initialize(system, dt, width, alpha, seed):
    """Create every trainable tensor from PyTorch's prescribed seeded RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        before = torch.get_rng_state().clone()
        model = CarrierScratchMap(system, dt, width, alpha)
        model.initial_rng_before = before
        model.initial_rng_after = torch.get_rng_state().clone()
        return model


def real_parameters(model):
    return sum(parameter.numel()*(2 if parameter.is_complex() else 1)
               for parameter in model.parameters())
