"""Environment poisoning (Novelty 2): a_poison = -a + eps, r_poison = -r."""
import numpy as np


def build_poison_buffer(buf, cfg, rng):
    idx = np.array(buf.idx_by_sector[cfg.unlearn_sector], dtype=np.int64)
    n = len(idx)
    a_p = np.clip(-buf.a[idx] + rng.normal(0.0, cfg.poison_noise, (n, cfg.action_dim)), -1, 1)
    return {
        "s": buf.s[idx].copy(),
        "a": a_p.astype(np.float32),
        "r": (-buf.r[idx]).copy(),
        "s2": buf.s2[idx].copy(),
        "d": buf.d[idx].copy(),
    }


def sample_poison(pb, batch, device, rng):
    n = len(pb["s"])
    idx = rng.integers(0, n, batch)
    import torch
    t = lambda x: torch.as_tensor(x[idx], device=device)
    return {"s": t(pb["s"]), "a": t(pb["a"]), "r": t(pb["r"]),
            "s2": t(pb["s2"]), "d": t(pb["d"])}
