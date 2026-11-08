"""Original sequence loss, balanced sampler and RNG state operations."""
import random
import numpy as np
import torch

def sequence_loss(model, samples, steps):
    """Free autoregression from true t=0; arithmetic mean over EVERY native step."""
    state = samples[:, 0]
    losses = []
    for index in range(1, steps + 1):
        state, _ = model.step(state)
        losses.append(((state - samples[:, index]) / model.scale).square().mean())
    return torch.stack(losses).mean()


def sampling(labels, seed, updates):
    """Same seed/ranks/function IDs/window schedule for all three architectures."""
    rng = np.random.default_rng(26101550 + seed)
    smooth = np.flatnonzero(labels == "smooth")
    discontinuous = np.flatnonzero(labels == "discontinuous")
    if not len(smooth) or not len(discontinuous):
        raise ValueError("Balanced sampler requires both families")
    ranks = rng.random((updates, 2, 8))
    identities = np.concatenate((smooth[(ranks[:, 0] * len(smooth)).astype(int)],
                                 discontinuous[(ranks[:, 1] * len(discontinuous)).astype(int)]), axis=1)
    endpoints = np.resize(np.array([4, 8, 16]), updates)
    return identities, endpoints, rng.bit_generator.state


def rng_state():
    return {"torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(), "python": random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
