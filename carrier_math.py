"""Unchanged characteristic/SV formulas, portable import wiring."""
import math
import torch

def characteristic_speed(state, system):
    """Return max mean-state characteristic speed |v|+c for each function."""
    mean = state.mean(-1)
    density = mean[:, 0]
    velocity = mean[:, 1] / density
    if system == "swe":
        sound = density.sqrt()
    else:
        pressure = .4 * (mean[:, 2] - .5 * mean[:, 1].square() / density)
        sound = (1.4 * pressure / density).sqrt()
    return velocity.abs() + sound


def multiplier(state, system, dt, alpha, power):
    """Prescribed scalar exp(-dt*mu_N) for every nonnegative Fourier mode."""
    cells = state.shape[-1]
    cutoff, nyquist = math.ceil(math.sqrt(cells)), cells / 2
    if cutoff >= nyquist or dt <= 0 or alpha < 0 or power < 1:
        raise ValueError("Invalid SV grid or parameters")
    k = torch.arange(cells // 2 + 1, dtype=state.dtype, device=state.device)
    q = ((k-cutoff)/(nyquist-cutoff)).clamp(0, 1).pow(power)
    rate = alpha * characteristic_speed(state, system)[:, None] / cells
    return torch.exp(-dt * rate * (2*torch.pi*k).square()[None, :] * q[None, :])


def filtered_carrier(u, system, dt, alpha, power):
    """Fuse scalar damping with characteristic translation at the conserved mean.

    The real even-grid Nyquist mode retains the inherited real-part convention.
    Coefficient extraction and right eigenvectors match the historical carrier.
    """
    mean = u.mean(-1, keepdim=True)
    perturbation = u-mean
    rho, velocity = mean[:, 0, 0], mean[:, 1, 0]/mean[:, 0, 0]
    if system == "swe":
        sound = rho.sqrt()
        minus = ((velocity+sound)[:, None]*perturbation[:, 0]-perturbation[:, 1])/(2*sound[:, None])
        plus = perturbation[:, 0]-minus
        coefficients = [minus, plus]
        vectors = [torch.stack((torch.ones_like(rho), velocity-sound), 1),
                   torch.stack((torch.ones_like(rho), velocity+sound), 1)]
        speeds = [velocity-sound, velocity+sound]
    else:
        pressure = .4*(mean[:, 2, 0]-.5*rho*velocity**2)
        sound2 = 1.4*pressure/rho
        sound = sound2.sqrt()
        enthalpy = (mean[:, 2, 0]+pressure)/rho
        pressure_delta = .4*(perturbation[:, 2]-velocity[:, None]*perturbation[:, 1]
                             +.5*velocity[:, None]**2*perturbation[:, 0])
        momentum_delta = perturbation[:, 1]-velocity[:, None]*perturbation[:, 0]
        minus = (pressure_delta-sound[:, None]*momentum_delta)/(2*sound2[:, None])
        plus = (pressure_delta+sound[:, None]*momentum_delta)/(2*sound2[:, None])
        contact = perturbation[:, 0]-pressure_delta/sound2[:, None]
        coefficients = [minus, contact, plus]
        vectors = [torch.stack((torch.ones_like(rho), velocity-sound, enthalpy-velocity*sound), 1),
                   torch.stack((torch.ones_like(rho), velocity, .5*velocity**2), 1),
                   torch.stack((torch.ones_like(rho), velocity+sound, enthalpy+velocity*sound), 1)]
        speeds = [velocity-sound, velocity, velocity+sound]
    cells = u.shape[-1]
    k = torch.arange(cells//2+1, dtype=u.dtype, device=u.device)
    damping = multiplier(u, system, dt, alpha, power)
    result = mean.expand_as(u).clone()
    for coefficient, vector, speed in zip(coefficients, vectors, speeds):
        phase = torch.exp(-2j*torch.pi*speed[:, None]*dt*k[None, :])
        transported = torch.fft.irfft(torch.fft.rfft(coefficient)*phase*damping, n=cells)
        result = result + vector[:, :, None]*transported[:, None, :]
    return result-result.mean(-1, keepdim=True)+mean
