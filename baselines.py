"""Independent periodic 1D FNO and U-Net baselines at a fixed native interval.

Both native architectures map S-scaled conservative state directly to the next
S-scaled state. They contain no carrier, dyadic cancellation, mean projection,
spectral viscosity, positivity repair, or hidden clipping. The separately named
residual FNO physical-wrapper control interprets its learned output as an
increment rate, V=U+dt*P0*N(U), and applies the existing unchanged radial correction.
Its inference frame is selected by the runner. This residual control is
separately named and is not the native direct-next-state FNO.

Architecture sources: FNO https://arxiv.org/abs/2010.08895 and U-Net
https://doi.org/10.1007/978-3-319-24574-4_28. The U-Net is an explicit 1D
periodic regression adaptation, not the published segmentation implementation.
"""

import torch
from torch import nn
import torch.nn.functional as functional


def load_original_models():
    """Return the shared, unchanged repair implementation."""
    from . import models_core
    return models_core


class SpectralConvolution1D(nn.Module):
    """Learn all complex channel couplings for modes 0 through modes-1."""

    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat) / width)

    def forward(self, field):
        spectrum = torch.fft.rfft(field, dim=-1)
        modes = min(self.modes, spectrum.shape[-1])
        output = torch.zeros_like(spectrum)
        output[..., :modes] = torch.einsum(
            "bik,iok->bok", spectrum[..., :modes], self.weight[..., :modes])
        return torch.fft.irfft(output, n=field.shape[-1], dim=-1)


class FNO1D(nn.Module):
    """Four-block autoregressive FNO with a two-layer pointwise projection.

    Inputs and outputs have shape (functions, conservative components, cells).
    Domain length is one, with a periodic grid. No coordinate channel is needed
    on this translation-homogeneous domain. Width44 matches the wide D-CINO
    real-parameter budget within a few percent. All weights are random tensors.
    """

    def __init__(self, system, width=44, modes=16, depth=4, dt=.02, physical_wrapper=False):
        super().__init__()
        if system not in ("swe", "euler") or width < 1 or depth < 1 or modes < 1:
            raise ValueError("Invalid system or FNO dimensions")
        channels = 2 if system == "swe" else 3
        self.system, self.dt = system, dt
        self.physical_wrapper = physical_wrapper
        self.register_buffer("scale", torch.tensor(
            [1., 1.] if system == "swe" else [1., 1., 2.5]).reshape(1, -1, 1))
        self.lift = nn.Conv1d(channels, width, 1)
        self.spectral = nn.ModuleList([SpectralConvolution1D(width, modes) for _ in range(depth)])
        self.pointwise = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(depth)])
        self.projection = nn.Sequential(nn.Conv1d(width, 2*width, 1), nn.GELU(),
                                        nn.Conv1d(2*width, channels, 1))
        self.original = load_original_models() if physical_wrapper else None

    def forward(self, state):
        latent = self.lift(state / self.scale)
        for index, (spectral, pointwise) in enumerate(zip(self.spectral, self.pointwise)):
            latent = spectral(latent) + pointwise(latent)
            if index < len(self.spectral)-1:
                latent = functional.gelu(latent)
        return self.scale * self.projection(latent)

    def step(self, state):
        candidate = self(state)
        if self.physical_wrapper:
            candidate = state + self.dt*(candidate-candidate.mean(dim=-1, keepdim=True))
            return self.original.safety(state, candidate, self.system)
        return candidate, torch.ones(state.shape[0], dtype=state.dtype, device=state.device)


class PeriodicDoubleConvolution(nn.Module):
    """Two circular three-tap convolutions and the original U-Net ReLU choice."""

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, 3, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv1d(output_channels, output_channels, 3, padding=1, padding_mode="circular"),
            nn.ReLU())

    def forward(self, field):
        return self.layers(field)


class PeriodicUNet1D(nn.Module):
    """Three-level periodic regression U-Net with concatenated encoder skips.

    Adaptations are 1D circular convolutions, average pooling, nearest-neighbor
    upsampling with pointwise channel reduction, same-size output and fixed-S
    physical channels. No normalization, dropout, repair or output constraint is
    applied. Base width21 matches the existing wide D-CINO parameter budget.
    """

    def __init__(self, system, width=21, dt=.02):
        super().__init__()
        if system not in ("swe", "euler") or width < 1:
            raise ValueError("Invalid system or U-Net width")
        channels = 2 if system == "swe" else 3
        self.system, self.dt = system, dt
        self.register_buffer("scale", torch.tensor(
            [1., 1.] if system == "swe" else [1., 1., 2.5]).reshape(1, -1, 1))
        self.encoder = nn.ModuleList([
            PeriodicDoubleConvolution(channels, width),
            PeriodicDoubleConvolution(width, 2*width),
            PeriodicDoubleConvolution(2*width, 4*width)])
        self.bottleneck = PeriodicDoubleConvolution(4*width, 8*width)
        self.reduce = nn.ModuleList([nn.Conv1d(8*width, 4*width, 1),
                                     nn.Conv1d(4*width, 2*width, 1),
                                     nn.Conv1d(2*width, width, 1)])
        self.decoder = nn.ModuleList([
            PeriodicDoubleConvolution(8*width, 4*width),
            PeriodicDoubleConvolution(4*width, 2*width),
            PeriodicDoubleConvolution(2*width, width)])
        self.projection = nn.Conv1d(width, channels, 1)

    def forward(self, state):
        if state.shape[-1] % 8:
            raise ValueError("Three U-Net pooling levels require a cell count divisible by eight")
        latent, skips = state/self.scale, []
        for encoder in self.encoder:
            latent = encoder(latent)
            skips.append(latent)
            latent = functional.avg_pool1d(latent, 2)
        latent = self.bottleneck(latent)
        for reduction, decoder, skip in zip(self.reduce, self.decoder, reversed(skips)):
            latent = reduction(functional.interpolate(latent, size=skip.shape[-1], mode="nearest"))
            latent = decoder(torch.cat((skip, latent), dim=1))
        return self.scale*self.projection(latent)

    def step(self, state):
        return self(state), torch.ones(state.shape[0], dtype=state.dtype, device=state.device)


def initialize(method, system, seed, **options):
    """Initialize one independent model from the prescribed CPU RNG seed."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        rng_before = torch.get_rng_state().clone()
        if method == "unet":
            model = PeriodicUNet1D(system, **options)
        else:
            if method not in ("fno", "fno_physical_wrapper"):
                raise ValueError("Unknown baseline method")
            model = FNO1D(system, physical_wrapper=method == "fno_physical_wrapper", **options)
        model.initial_rng_before = rng_before
        model.initial_rng_after = torch.get_rng_state().clone()
        return model


def real_parameters(model):
    """Count real scalar degrees of freedom; one complex scalar counts as two."""
    return sum(parameter.numel()*(2 if parameter.is_complex() else 1)
               for parameter in model.parameters())
