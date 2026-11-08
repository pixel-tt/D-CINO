"""Radial spectral viscosity on the unsplit 2D SWE dyadic flow map.

No learned physical mixing or directional characteristic-box projection is used.
The inherited positivity/total-entropy ray repair is applied after the candidate.
"""
import math
import torch



def radial_multiplier(u, dt=.02, alpha=.05, power=2):
    """Return exp[-dt alpha c_A h (2 pi |k|)^2 q_N(k)].

    Square periodic grids represent [0,1)^2. h=1/N, m_N=ceil(sqrt(N)),
    q_N=clip((|k|-m_N)/(N/2-m_N),0,1)^power. c_A is the mean-state
    velocity magnitude plus sqrt(mean depth). The same real multiplier acts
    on h, hu and hv; zero and low frequencies are exactly unchanged.
    """
    n, ny = u.shape[-2:]
    cutoff = math.ceil(math.sqrt(n))
    if n != ny or n / 2 <= cutoff:
        raise ValueError('Square grids with N/2 > ceil(sqrt(N)) required')
    if alpha < 0 or dt < 0 or power <= 0:
        raise ValueError('alpha, dt must be nonnegative and power positive')
    mean = u.mean((-2, -1), keepdim=True)
    if not bool(torch.isfinite(mean).all() & (mean[:, :1] > 0).all()):
        raise ValueError('Finite positive mean depth required')
    # norm has a defined zero derivative; sqrt(sum(v^2)) does not at v=0.
    speed = torch.linalg.vector_norm(mean[:, 1:] / mean[:, :1], dim=1, keepdim=True)
    c_a = speed + mean[:, :1].sqrt()
    k = torch.fft.fftfreq(n, d=1 / n, device=u.device, dtype=u.dtype)
    kx, ky = torch.meshgrid(k, k, indexing='ij')
    radius = torch.sqrt(kx.square() + ky.square())
    profile = ((radius - cutoff) / (n / 2 - cutoff)).clamp(0, 1).pow(power)
    rate = alpha * c_a / n * (2 * torch.pi * radius).square() * profile
    return torch.exp(-dt * rate)
