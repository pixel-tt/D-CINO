"""Independent periodic 2D FNO and U-Net regression baselines.

FNO follows Li et al. (arXiv:2010.08895): four Fourier/pointwise blocks,
GELU except after the final block, then a pointwise projection. U-Net follows
Ronneberger et al. (doi:10.1007/978-3-319-24574-4_28), adapted to periodic
regression with circular padded convolutions, three pooling levels, learned
transposed-convolution upsampling, skip concatenation, and no normalization.
Both predict the full next conservative state directly. There is no residual
addition, characteristic carrier, dyadic operation, mean correction, or Q in
the native baselines. The separate wrapped FNO learns a residual increment
U + dt*P0*N(U), with conservative scaling S=I, then applies Q. This is an
explicit structural control, not an unmodified standard FNO baseline.
"""
import torch
from torch import nn
from torch.nn import functional as F


class FourierBlock2d(nn.Module):
    """Learn the two low-frequency slabs of the real 2D Fourier transform."""
    def __init__(self, width=24, modes=4):
        super().__init__()
        self.modes = modes
        self.weights = nn.Parameter(torch.randn(2, width, width, modes, modes,
                                                dtype=torch.cfloat) / width)
        self.pointwise = nn.Conv2d(width, width, 1)

    def forward(self, features):
        transformed = torch.fft.rfft2(features)
        output = torch.zeros_like(transformed)
        modes = min(self.modes, features.shape[-2] // 2, transformed.shape[-1])
        output[:, :, :modes, :modes] = torch.einsum(
            'bixy,ioxy->boxy', transformed[:, :, :modes, :modes],
            self.weights[0, :, :, :modes, :modes])
        output[:, :, -modes:, :modes] = torch.einsum(
            'bixy,ioxy->boxy', transformed[:, :, -modes:, :modes],
            self.weights[1, :, :, :modes, :modes])
        return torch.fft.irfft2(output, s=features.shape[-2:]) + self.pointwise(features)


class FNO2d(nn.Module):
    """Full next-state periodic FNO; coordinate channels omitted for the torus."""
    def __init__(self, width=24, modes=4, depth=4, dt=.02, wrapped=False):
        super().__init__()
        self.dt, self.wrapped = dt, wrapped
        self.lift = nn.Conv2d(3, width, 1)
        self.blocks = nn.ModuleList([FourierBlock2d(width, modes) for _ in range(depth)])
        self.project = nn.Sequential(nn.Conv2d(width, 2 * width, 1), nn.GELU(),
                                     nn.Conv2d(2 * width, 3, 1))

    def forward(self, state):
        features = self.lift(state)
        for index, block in enumerate(self.blocks):
            features = block(features)
            if index + 1 < len(self.blocks):
                features = F.gelu(features)
        return self.project(features)

    def step(self, state):
        candidate = self(state)
        if self.wrapped:
            from .swe2d import safety
            # Fixed S=I; P0 removes the spatial mean of the learned increment.
            increment = candidate - candidate.mean((-2, -1), keepdim=True)
            candidate = state + self.dt * increment
            return safety(state, candidate)
        return candidate, torch.ones(len(state), dtype=state.dtype, device=state.device)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode='circular'),
            nn.ReLU())

    def forward(self, features):
        return self.layers(features)


class PeriodicUNet2d(nn.Module):
    """Three-level periodic U-Net producing the full next conservative state."""
    def __init__(self, width=9, dt=.02):
        super().__init__()
        self.dt, self.wrapped = dt, False
        widths = [width * 2 ** index for index in range(4)]
        self.encoder = nn.ModuleList([DoubleConv(3, widths[0])]
                                    + [DoubleConv(widths[index - 1], widths[index])
                                       for index in range(1, 4)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(widths[index], widths[index - 1], 2, stride=2)
                                 for index in (3, 2, 1)])
        self.decoder = nn.ModuleList([DoubleConv(2 * widths[index - 1], widths[index - 1])
                                     for index in (3, 2, 1)])
        self.project = nn.Conv2d(widths[0], 3, 1)

    def forward(self, state):
        features, skips = state, []
        for index, encoder in enumerate(self.encoder):
            if index:
                features = F.max_pool2d(features, 2)
            features = encoder(features)
            skips.append(features)
        for up, decoder, skip in zip(self.up, self.decoder, reversed(skips[:-1])):
            features = decoder(torch.cat((up(features), skip), dim=1))
        return self.project(features)

    def step(self, state):
        return self(state), torch.ones(len(state), dtype=state.dtype, device=state.device)


def build_model(method, width=None, dt=.02):
    if method in ('fno_native', 'fno_physical_wrapper'):
        return FNO2d(width=24 if width is None else width, dt=dt,
                     wrapped=method == 'fno_physical_wrapper')
    if method == 'unet_native':
        return PeriodicUNet2d(width=9 if width is None else width, dt=dt)
    raise ValueError('Unknown baseline: ' + method)


def parameter_count(model):
    """Count real scalars, including real and imaginary Fourier parameters."""
    return sum(parameter.numel() * (2 if parameter.is_complex() else 1)
               for parameter in model.parameters())
