"""Spectral/local neural flow maps and the persistent-transport construction.

All neural tensor layouts are (batch, components, cells). Fixed-budget training
uses the raw conservative residual; safety is an explicitly evaluated inference
map. This is not an implementation of the published LGNO or PACE-FNO packages.
"""
import torch
from torch import nn
import torch.nn.functional as functional


class SpectralLayer(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        self.weights = nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat) / width)

    def forward(self, u):
        freq = torch.fft.rfft(u)
        output = torch.zeros_like(freq)
        modes = min(self.modes, freq.shape[-1])
        output[..., :modes] = torch.einsum("bik,iok->bok", freq[..., :modes], self.weights[..., :modes])
        return torch.fft.irfft(output, n=u.shape[-1])


class PeriodicStencil(nn.Module):
    """Original circular Conv1d, evaluated as the same explicit five-tap sum."""
    def __init__(self, width):
        super().__init__()
        layer=nn.Conv1d(width,width,5,padding=2,padding_mode="circular")
        self.weight,self.bias=layer.weight,layer.bias
    def forward(self,z):
        patches=torch.stack([torch.roll(z,2-j,dims=-1) for j in range(5)],dim=2)
        return torch.matmul(self.weight.reshape(self.weight.shape[0],-1),patches.reshape(z.shape[0],-1,z.shape[-1]))+self.bias[None,:,None]

class FlowMap(nn.Module):
    """Residual Fourier flow map with an optional circular local-convolution path.

    The zero-mode correction enforces global conservative-component means. Fixed
    nondimensional scales are prescribed, not estimated from held-out fields.
    Parameters control only the nonlinear interaction map at the trained dt.
    """
    def __init__(self, system, width=24, modes=16, depth=3, local=True, characteristic=False, quadratic=False, dt=0.02, normalized=False, dyadic=False):
        super().__init__()
        channels = 2 if system == "swe" else 3
        self.system = system
        self.characteristic = characteristic
        self.quadratic = quadratic
        self.dt = dt
        self.normalized = normalized
        self.dyadic = dyadic
        self.register_buffer("scale", torch.tensor([1., 1.] if system == "swe" else [1., 1., 2.5]).reshape(1, -1, 1))
        self.lift = nn.Conv1d(channels, width, 1)
        self.spectral = nn.ModuleList([SpectralLayer(width, modes) for _ in range(depth)])
        self.pointwise = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(depth)])
        self.local = nn.ModuleList([PeriodicStencil(width) for _ in range(depth)]) if local else None
        self.output = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, channels, 1))

    def forward(self, u):
        mean = u.mean(dim=-1, keepdim=True)
        fluctuation = (u-mean)/self.scale
        amplitude = fluctuation.square().mean(dim=(1, 2), keepdim=True)
        # Clamp before sqrt: identical forward normalization, finite input JVP at constants.
        features = mean/self.scale + fluctuation/torch.sqrt(amplitude.clamp_min(1e-16)) if self.normalized else u/self.scale
        if self.dyadic:
            features = torch.cat((u/self.scale, (u+mean)/(2*self.scale)), dim=0)
        z = self.lift(features)
        for k, (spectral, pointwise) in enumerate(zip(self.spectral, self.pointwise)):
            value = spectral(z) + pointwise(z)
            if self.local is not None:
                value = value + self.local[k](z)
            z = functional.gelu(value)
        gate = amplitude if self.quadratic else 0.2
        neural = self.output(z)
        if self.dyadic:
            full, half = neural.chunk(2, dim=0)
            neural = 2*(full-2*half)
        increment = gate*self.scale*neural
        base = characteristic_propagate(u, self.system, self.dt) if self.characteristic else u
        return base + increment - increment.mean(dim=-1, keepdim=True)


def characteristic_propagate(u, system, dt):
    """Exact Fourier propagation for the physical Jacobian at the conserved mean.

    The real even-grid Nyquist coefficient uses the real part of its phase,
    hence is contractive rather than an invertible shift. Exact group/isometry
    statements require odd grids or zero Nyquist content. Other modes are exact.
    The nonlinear Euler/SWE solution is not equal to this linearized map.
    """
    mean = u.mean(dim=-1, keepdim=True)
    perturbation = u-mean
    rho, velocity = mean[:, 0, 0], mean[:, 1, 0]/mean[:, 0, 0]
    if system == "swe":
        sound = torch.sqrt(rho)
        minus = ((velocity+sound)[:, None]*perturbation[:, 0]-perturbation[:, 1])/(2*sound[:, None])
        plus = perturbation[:, 0]-minus
        right_minus = torch.stack((torch.ones_like(rho), velocity-sound), dim=1)
        right_plus = torch.stack((torch.ones_like(rho), velocity+sound), dim=1)
        coefficients = [minus, plus]
        vectors = [right_minus, right_plus]
        speeds = [velocity-sound, velocity+sound]
    else:
        pressure = 0.4*(mean[:, 2, 0]-0.5*rho*velocity**2)
        sound2 = 1.4*pressure/rho
        sound = torch.sqrt(sound2)
        enthalpy = (mean[:, 2, 0]+pressure)/rho
        pressure_delta = 0.4*(perturbation[:, 2]-velocity[:, None]*perturbation[:, 1]+0.5*velocity[:, None]**2*perturbation[:, 0])
        momentum_delta = perturbation[:, 1]-velocity[:, None]*perturbation[:, 0]
        minus = (pressure_delta-sound[:, None]*momentum_delta)/(2*sound2[:, None])
        plus = (pressure_delta+sound[:, None]*momentum_delta)/(2*sound2[:, None])
        contact = perturbation[:, 0]-pressure_delta/sound2[:, None]
        coefficients = [minus, contact, plus]
        vectors = [torch.stack((torch.ones_like(rho), velocity-sound, enthalpy-velocity*sound), dim=1),
                   torch.stack((torch.ones_like(rho), velocity, 0.5*velocity**2), dim=1),
                   torch.stack((torch.ones_like(rho), velocity+sound, enthalpy+velocity*sound), dim=1)]
        speeds = [velocity-sound, velocity, velocity+sound]
    result = mean.expand_as(u).clone()
    for coefficient, vector, speed in zip(coefficients, vectors, speeds):
        transported = translate(coefficient[:, None, :], speed*dt*u.shape[-1], "spectral")
        result = result+vector[:, :, None]*transported
    return result


def galilean(u, v, system):
    v = v.reshape(-1, 1)
    components = [u[:, 0], u[:, 1] + v*u[:, 0]]
    if system == "euler":
        components.append(u[:, 2]+v*u[:, 1]+0.5*v**2*u[:, 0])
    return torch.stack(components, dim=1)


def primitive(u, system):
    rho = u[:, 0]
    vel = u[:, 1]/rho
    if system == "swe":
        return torch.stack((rho, vel), dim=1)
    return torch.stack((rho, vel, 0.4*(u[:, 2]-0.5*u[:, 1]*vel)), dim=1)


def entropy(u, system):
    rho = u[:, 0]
    if system == "swe":
        return 0.5*u[:, 1]**2/rho + 0.5*rho**2
    pressure = 0.4*(u[:, 2]-0.5*u[:, 1]**2/rho)
    return -rho*(torch.log(pressure.clamp_min(1e-30))-1.4*torch.log(rho.clamp_min(1e-30)))/0.4


def safety(u, candidate, system, iterations=24, floor=1e-7):
    """Ray retraction toward the conservative mean with positivity and total entropy.

    In exact arithmetic the feasible interval contains zero by convexity.
    Bisection retains the feasible endpoint. The density/pressure floor and
    floating-point entropy tolerance are operational, not exact real guarantees.
    Returns corrected field and per-function retained amplitude.
    """
    mean = u.mean(dim=-1, keepdim=True)
    candidate = candidate-candidate.mean(dim=-1, keepdim=True)+mean
    direction = candidate-mean
    target = entropy(u, system).mean(dim=-1)
    tolerance = 8*torch.finfo(u.dtype).eps*(1+target.abs())

    def feasible(theta):
        state = mean + theta[:, None, None]*direction
        rho_ok = state[:, 0].amin(dim=-1) >= floor
        internal_ok = torch.ones_like(rho_ok)
        if system == "euler":
            internal = state[:, 2]-0.5*state[:, 1]**2/state[:, 0].clamp_min(floor)
            internal_ok = internal.amin(dim=-1) >= floor/0.4
        entropy_ok = entropy(state, system).mean(dim=-1) <= target+tolerance
        return rho_ok & internal_ok & entropy_ok & torch.isfinite(state).all(dim=(1, 2))

    full = torch.ones(u.shape[0], dtype=u.dtype, device=u.device)
    valid = feasible(full)
    if bool(valid.all()):
        return candidate, full
    lower, upper = torch.zeros_like(full), full
    for _ in range(iterations):
        middle = (lower+upper)*0.5
        ok = feasible(middle)
        lower = torch.where(ok, middle, lower)
        upper = torch.where(ok, upper, middle)
    theta = torch.where(valid, full, lower)
    return mean + theta[:, None, None]*direction, theta


def translate(u, shifts, kind="linear"):
    """Periodic translation by cell-valued shifts; linear remap is a Markov map."""
    n = u.shape[-1]
    if kind == "spectral":
        k = torch.arange(n//2+1, device=u.device, dtype=u.dtype)
        phase = torch.exp(-2j*torch.pi*shifts[:, None, None]*k[None, None, :]/n)
        return torch.fft.irfft(torch.fft.rfft(u)*phase, n=n)
    integer = torch.floor(shifts).long()
    fraction = (shifts-integer)[:, None, None]
    indices = (torch.arange(n, device=u.device)[None, :]-integer[:, None]) % n
    indices = indices[:, None, :].expand(-1, u.shape[1], -1)
    return (1-fraction)*torch.gather(u, 2, indices)+fraction*torch.gather(u, 2, (indices-1) % n)
