"""Periodic 2D SWE: unsplit MUSCL-HLL/SSPRK2 reference and analytic carrier.

States are (batch, 3, nx, ny), components (h, hu, hv), g=1. All reference
arithmetic is float64. Reconstruction is in primitive variables with MC slopes.
The reference is an empirical approximation, not an entropy-stability theorem.
"""
import time
import numpy as np
import torch


def primitive(u):
    return torch.cat((u[:, :1], u[:, 1:] / u[:, :1]), 1)


def conservative(q):
    return torch.cat((q[:, :1], q[:, 1:] * q[:, :1]), 1)


def entropy(u):
    return 0.5 * u[:, 1:].square().sum(1) / u[:, 0] + 0.5 * u[:, 0].square()


def galilean(u, velocity):
    return torch.cat((u[:, :1], u[:, 1:] + velocity[:, :, None, None] * u[:, :1]), 1)


def mean_velocity(u):
    return u[:, 1:].sum((-2, -1)) / u[:, :1].sum((-2, -1))


def restrict(u, n):
    nx, ny = u.shape[-2:]
    if nx % n or ny % n:
        raise ValueError('Reference grid must be divisible by coarse grid')
    return u.reshape(*u.shape[:-2], n, nx // n, n, ny // n).mean((-3, -1))


def minmod(a, b, c):
    same = (a.sign() == b.sign()) & (b.sign() == c.sign())
    return torch.where(same, a.sign() * torch.minimum(a.abs(), torch.minimum(b.abs(), c.abs())), 0.)


def flux(q, axis):
    u = conservative(q)
    vel = q[:, axis + 1:axis + 2]
    f = u * vel
    f[:, axis + 1] += 0.5 * q[:, 0].square()
    sound = q[:, :1].sqrt()
    return u, f, vel - sound, vel + sound


def residual(u):
    q = primitive(u)
    result = torch.zeros_like(u)
    for axis in (0, 1):
        dim = axis + 2
        left = q - q.roll(1, dim)
        right = q.roll(-1, dim) - q
        slope = minmod(2 * left, (left + right) / 2, 2 * right)
        ul, fl, ll, lr = flux(q + slope / 2, axis)
        ur, fr, rl, rr = flux((q - slope / 2).roll(-1, dim), axis)
        sl = torch.minimum(ll, rl).clamp_max(0)
        sr = torch.maximum(lr, rr).clamp_min(0)
        f = (sr * fl - sl * fr + sl * sr * (ur - ul)) / (sr - sl).clamp_min(1e-14)
        result -= u.shape[dim] * (f - f.roll(1, dim))
    return result


def admissible(u):
    return bool(torch.isfinite(u).all() & (u[:, 0].amin() > 1e-10))


@torch.no_grad()
def solve(initial, times, cfl=0.35):
    """Unsplit spatial divergence, SSPRK2, combined x/y CFL; reject bad stages."""
    u = initial.to(torch.float64).clone()
    if not admissible(u):
        raise ValueError('Invalid initial state')
    initial_mean = u.mean((-2, -1))
    e0 = entropy(u).mean((-2, -1))
    snapshots, t, steps, retries = [u.clone()], 0., 0, 0
    start = time.perf_counter()
    max_entropy_step = 0.
    for target in times:
        while t < target - 1e-13:
            q = primitive(u)
            c = q[:, 0].sqrt()
            rate = u.shape[-2] * (q[:, 1].abs() + c) + u.shape[-1] * (q[:, 2].abs() + c)
            dt = min(cfl / float(rate.max()), float(target) - t)
            for _ in range(16):
                stage = u + dt * residual(u)
                if admissible(stage):
                    candidate = (u + stage + dt * residual(stage)) / 2
                    if admissible(candidate):
                        break
                retries += 1
                dt /= 2
            else:
                raise FloatingPointError('SSPRK2 positivity retries exhausted')
            max_entropy_step = max(max_entropy_step, float((entropy(candidate).mean((-2,-1)) - entropy(u).mean((-2,-1))).max()))
            u = candidate
            t += dt
            steps += 1
        snapshots.append(u.clone())
    if u.is_cuda:
        torch.cuda.synchronize()
    return torch.stack(snapshots, 1), dict(steps=steps, rejected_steps=retries,
        seconds=time.perf_counter()-start, conservation_linf=float((u.mean((-2,-1))-initial_mean).abs().max()),
        min_h=float(u[:,0].min()), max_entropy_change=float((entropy(u).mean((-2,-1))-e0).max()),
        max_entropy_step=max_entropy_step)


def carrier(u, dt, multiplier=None):
    """exp[-2 pi i dt (kx Jx + ky Jy)] at the conservative mean.

    Resolve longitudinal acoustic and transverse shear modes jointly. Full FFT
    followed by real projection handles even-grid Nyquist planes contractively;
    odd grids retain the exact real linear group. No directional splitting.
    """
    mean = u.mean((-2, -1), keepdim=True)
    h = mean[:, :1]
    vel = mean[:, 1:] / h
    c = h.sqrt()
    w = u - mean
    f = torch.fft.fft2(w)
    dh = f[:, :1]
    j = f[:, 1:] - vel * dh
    nx, ny = u.shape[-2:]
    kx = torch.fft.fftfreq(nx, d=1/nx, device=u.device, dtype=u.dtype)[:, None]
    ky = torch.fft.fftfreq(ny, d=1/ny, device=u.device, dtype=u.dtype)[None, :]
    radius = (kx.square() + ky.square()).sqrt()
    direction = torch.stack((kx.expand(nx,ny), ky.expand(nx,ny)), 0) / radius.clamp_min(1)
    longitudinal = (j * direction).sum(1, keepdim=True)
    angle = 2 * torch.pi * dt * c * radius
    phase = torch.exp(-2j * torch.pi * dt * (vel * torch.stack((kx.expand(nx,ny),ky.expand(nx,ny)),0)).sum(1,keepdim=True))
    out_h = phase * (dh * angle.cos() - 1j * longitudinal / c * angle.sin())
    out_l = phase * (longitudinal * angle.cos() - 1j * c * dh * angle.sin())
    out_j = phase * (j - direction * longitudinal) + direction * out_l
    transformed = torch.cat((out_h, out_j + vel * out_h), 1)
    if multiplier is not None:
        transformed = transformed * multiplier
    return mean + torch.fft.ifft2(transformed).real


def safety(u, candidate, iterations=24, floor=1e-7):
    """Mean-preserving ray retraction enforcing h floor and total convex entropy."""
    mean = u.mean((-2,-1), keepdim=True)
    candidate = candidate - candidate.mean((-2,-1), keepdim=True) + mean
    direction = candidate - mean
    target = entropy(u).mean((-2,-1))
    tol = 8 * torch.finfo(u.dtype).eps * (1 + target.abs())
    def feasible(theta):
        q = mean + theta[:,None,None,None] * direction
        return (q[:,0].amin((-2,-1)) >= floor) & (entropy(q).mean((-2,-1)) <= target+tol) & torch.isfinite(q).flatten(1).all(1)
    full = torch.ones(len(u), device=u.device, dtype=u.dtype)
    valid = feasible(full)
    if bool(valid.all()):
        return candidate, full
    low, high = torch.zeros_like(full), full
    for _ in range(iterations):
        mid = (low+high)/2
        ok = feasible(mid)
        low, high = torch.where(ok,mid,low), torch.where(ok,high,mid)
    theta = torch.where(valid,full,low)
    return mean + theta[:,None,None,None]*direction, theta


def translate(u, shifts):
    """Positive bilinear periodic remap; shifts are (batch, x/y) in cells."""
    for axis in (0,1):
        dim = axis+2
        n = u.shape[dim]
        integer = shifts[:,axis].floor().long()
        fraction = (shifts[:,axis]-integer)[:,None,None,None]
        shape = (len(u),1,n,1) if axis == 0 else (len(u),1,1,n)
        index = ((torch.arange(n,device=u.device)[None,:]-integer[:,None]) % n).reshape(shape).expand_as(u)
        u = (1-fraction)*u.gather(dim,index) + fraction*u.gather(dim,(index-1)%n)
    return u


@torch.no_grad()
def rollout(model, initial, steps, dt, use_safety=True):
    velocity = mean_velocity(initial)
    current = galilean(initial,-velocity)
    out, retained = [initial], []
    for k in range(1,steps+1):
        candidate = model(current)
        if use_safety:
            current, theta = safety(current,candidate)
        else:
            current, theta = candidate, torch.ones(len(initial),device=initial.device)
        rendered = translate(current, velocity*(k*dt*initial.shape[-1]))
        out.append(galilean(rendered,velocity))
        retained.append(theta)
    return torch.stack(out,1), torch.stack(retained,1)


def initial_conditions(count, n, seed, amplitude=1., device='cpu'):
    """Independent functions: alternating nonparallel smooth modes / circular dams.

    Seed and index identify a function across grids. Fine-grid midpoint sampling
    is used; restriction creates the coarse finite-volume state. Mean velocity
    is removed once. No normalization is estimated from validation functions.
    """
    rng = np.random.default_rng(seed)
    x,y = np.meshgrid((np.arange(n)+.5)/n,(np.arange(n)+.5)/n,indexing='ij')
    fields, labels = [], []
    waves = [(1,0),(0,1),(1,1),(1,-1),(2,1),(1,2)]
    for i in range(count):
        hbar = rng.uniform(.9,1.2)
        q = np.zeros((3,n,n)); q[0] = hbar
        if i % 2 == 0:
            for kx,ky in waves:
                phases = rng.uniform(0,2*np.pi,3)
                amps = amplitude*.065*rng.uniform(-1,1,3)/np.sqrt(kx*kx+ky*ky)
                q += amps[:,None,None]*np.sin(2*np.pi*(kx*x+ky*y)[None]+phases[:,None,None])
            labels.append('smooth_nonparallel')
        else:
            cx,cy = rng.uniform(.3,.7,2)
            radius = rng.uniform(.12,.22)
            distance = np.minimum(abs(x-cx),1-abs(x-cx))**2+np.minimum(abs(y-cy),1-abs(y-cy))**2
            inside = distance < radius**2
            jump = rng.uniform(.25,.6)*amplitude
            q[0] += jump*(inside-inside.mean())
            speed = rng.uniform(-.12,.12,2)*amplitude
            q[1:] = speed[:,None,None]*(2*inside-1)
            labels.append('circular_dam')
        fields.append(q)
    q = torch.tensor(np.stack(fields),dtype=torch.float64,device=device)
    u = conservative(q)
    return galilean(u,-mean_velocity(u)), labels


def exact_shear(n, t, device='cpu'):
    """Exact cell averages of an oblique advected shear with constant depth."""
    x = (torch.arange(n,device=device,dtype=torch.float64)+.5)/n
    x,y = torch.meshgrid(x,x,indexing='ij')
    k = (1,2); mean = (.21,-.13); a = .13
    wave = a*np.sinc(k[0]/n)*np.sinc(k[1]/n)*torch.sin(2*torch.pi*(k[0]*x+k[1]*y-(k[0]*mean[0]+k[1]*mean[1])*t))
    return torch.stack((torch.ones_like(x),mean[0]-2/np.sqrt(5)*wave,mean[1]+1/np.sqrt(5)*wave),0)[None]
