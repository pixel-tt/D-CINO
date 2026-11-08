"""Transparent 1D finite-volume reference and conservative transport primitives.

States have shape (batch, cells, components). SWE uses (h, hu), g=1;
Euler uses (rho, rho*u, E), gamma=1.4. The reference is primitive-variable
MUSCL-HLL with MC slopes and SSPRK2, periodic boundaries, adaptive CFL 0.35.
Reference accuracy is empirical and requires a separate refinement check.
"""
from pathlib import Path
import argparse
import json
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GAMMA = 1.4


def primitive(state, system):
    density = state[..., 0]
    velocity = state[..., 1] / density
    if system == "swe":
        return np.stack((density, velocity), axis=-1)
    pressure = (GAMMA - 1) * (state[..., 2] - 0.5 * state[..., 1] * velocity)
    return np.stack((density, velocity, pressure), axis=-1)


def conservative(prim, system):
    density, velocity = prim[..., 0], prim[..., 1]
    if system == "swe":
        return np.stack((density, density * velocity), axis=-1)
    return np.stack((density, density * velocity,
                     prim[..., 2] / (GAMMA - 1) + 0.5 * density * velocity**2), axis=-1)


def galilean(state, velocity, system):
    """Apply G_v to conservative components, without translating spatial cells."""
    result = state.copy()
    v = np.asarray(velocity).reshape(-1, 1)
    result[..., 1] = state[..., 1] + v * state[..., 0]
    if system == "euler":
        result[..., 2] = state[..., 2] + v * state[..., 1] + 0.5 * v**2 * state[..., 0]
    return result


def mean_velocity(state):
    return state[..., 1].sum(axis=1) / state[..., 0].sum(axis=1)


def entropy(state, system):
    prim = primitive(state, system)
    if system == "swe":
        return 0.5 * state[..., 1] * prim[..., 1] + 0.5 * prim[..., 0]**2
    return -prim[..., 0] * np.log(prim[..., 2] / prim[..., 0]**GAMMA) / (GAMMA - 1)


def admissible(state, system, floor=1e-10):
    if not np.all(np.isfinite(state)) or np.min(state[..., 0]) <= floor:
        return False
    return system == "swe" or np.min(primitive(state, system)[..., 2]) > floor


def minmod3(a, b, c):
    same = (np.sign(a) == np.sign(b)) & (np.sign(a) == np.sign(c))
    return np.where(same, np.sign(a) * np.minimum(np.minimum(abs(a), abs(b)), abs(c)), 0.0)


def flux_and_speed(prim, system):
    state = conservative(prim, system)
    rho, vel = prim[..., 0], prim[..., 1]
    pressure = 0.5 * rho**2 if system == "swe" else prim[..., 2]
    flux = np.empty_like(state)
    flux[..., 0] = rho * vel
    flux[..., 1] = rho * vel**2 + pressure
    if system == "euler":
        flux[..., 2] = (state[..., 2] + pressure) * vel
    sound = np.sqrt(rho) if system == "swe" else np.sqrt(GAMMA * pressure / rho)
    return state, flux, vel - sound, vel + sound


def residual(state, system, order=2):
    prim = primitive(state, system)
    if order == 2:
        left = prim - np.roll(prim, 1, axis=1)
        right = np.roll(prim, -1, axis=1) - prim
        slope = minmod3(2 * left, 0.5 * (left + right), 2 * right)
    else:
        slope = np.zeros_like(prim)
    ul, fl, ll, lr = flux_and_speed(prim + 0.5 * slope, system)
    ur, fr, rl, rr = flux_and_speed(np.roll(prim - 0.5 * slope, -1, axis=1), system)
    sl = np.minimum(np.minimum(ll, rl), 0.0)[..., None]
    sr = np.maximum(np.maximum(lr, rr), 0.0)[..., None]
    flux = (sr * fl - sl * fr + sl * sr * (ur - ul)) / np.maximum(sr - sl, 1e-14)
    return -state.shape[1] * (flux - np.roll(flux, 1, axis=1))


def solve(state, system, output_times, cfl=0.35, order=2):
    """Integrate all samples to each output time; retry invalid steps by halving dt.

    A step rejection is numerical protection, not a proved MUSCL positivity claim.
    Returns cell-average snapshots and measured step/rejection/entropy diagnostics.
    """
    u = np.array(state, dtype=np.float64, copy=True)
    if not admissible(u, system):
        raise ValueError("Invalid initial state")
    outputs, t, steps, rejected = [u.copy()], 0.0, 0, 0
    initial_total = u.mean(axis=1)
    start = time.perf_counter()
    for target in output_times:
        while t < target - 1e-13:
            p = primitive(u, system)
            speed = abs(p[..., 1]) + (np.sqrt(p[..., 0]) if system == "swe" else np.sqrt(GAMMA*p[..., 2]/p[..., 0]))
            dt = min(cfl / (u.shape[1] * speed.max()), target-t)
            for retry in range(16):
                stage = u + dt * residual(u, system, order)
                if admissible(stage, system):
                    candidate = 0.5 * (u + stage + dt * residual(stage, system, order))
                    if admissible(candidate, system):
                        break
                rejected += 1
                dt *= 0.5
            else:
                raise FloatingPointError("SSPRK2 positivity retries exhausted")
            u = candidate
            t += dt
            steps += 1
        outputs.append(u.copy())
    return np.stack(outputs, axis=1), {"steps": steps, "rejected_steps": rejected,
            "elapsed_seconds": time.perf_counter()-start,
            "conservation_linf": float(abs(u.mean(axis=1)-initial_total).max()),
            "min_density": float(u[..., 0].min()),
            "max_entropy_change": float((entropy(u, system).mean(axis=1)-entropy(state, system).mean(axis=1)).max())}


def translate_linear(state, shifts):
    """Conservative positive periodic remap of piecewise-constant cell averages.

    shifts is measured in cells; integer shifts are exact permutations. The
    weights are shared by all conservative components and all cells of a sample.
    """
    shift = np.asarray(shifts).reshape(-1)
    n = state.shape[1]
    integer = np.floor(shift).astype(np.int64)
    fraction = (shift-integer)[:, None, None]
    indices = (np.arange(n)[None, :] - integer[:, None]) % n
    rows = np.arange(state.shape[0])[:, None]
    return (1-fraction)*state[rows, indices] + fraction*state[rows, (indices-1) % n]


def restrict(state, cells):
    factor = state.shape[-2] // cells
    if factor*cells != state.shape[-2]:
        raise ValueError("Reference grid must be divisible by coarse grid")
    return state.reshape(*state.shape[:-2], cells, factor, state.shape[-1]).mean(axis=-2)


def initial_conditions(count, cells, seed, system):
    """Generate independent positive smooth and four-state periodic fields.

    Function identity is the RNG seed and index, not a time window. A fine grid
    midpoint representation is refined independently in reference sensitivity.
    Global mass-weighted velocity is removed once from conservative averages.
    """
    rng = np.random.default_rng(seed)
    x = (np.arange(cells)+0.5)/cells
    channels = 2 if system == "swe" else 3
    fields, labels = [], []
    for i in range(count):
        base_rho = rng.uniform(0.8, 1.4)
        base_p = rng.uniform(0.6, 1.4)
        if i % 2 == 0:
            prim = np.zeros((cells, channels))
            prim[:, 0] = base_rho
            if system == "euler":
                prim[:, 2] = base_p
            for k in range(1, 5):
                phases = rng.uniform(0, 2*np.pi, channels)
                amplitudes = rng.uniform(-1, 1, channels) * 0.12 / k
                prim += amplitudes[None, :] * np.sin(2*np.pi*k*x[:, None]+phases[None, :])
            labels.append("smooth")
        else:
            offset = rng.uniform(0, 1)
            edges = np.array([0, 0.25, 0.5, 0.75])
            indices = np.minimum(np.searchsorted(edges, (x-offset) % 1, side="right")-1, 3)
            values = np.column_stack((rng.uniform(0.45, 1.8, 4), rng.uniform(-0.5, 0.5, 4)))
            if system == "euler":
                values = np.column_stack((values, rng.uniform(0.35, 1.8, 4)))
            prim = values[indices]
            labels.append("discontinuous")
        fields.append(conservative(prim, system))
    states = np.stack(fields)
    return galilean(states, -mean_velocity(states), system), np.array(labels)
