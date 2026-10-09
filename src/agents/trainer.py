"""Phase-1 training loop (algorithm-agnostic). Records per-episode summary
statistics x_t as (stat, sector) tuples for the binary-tree mechanism."""
import numpy as np
import torch

from .sac import SACAgent, ReplayBuffer
from .ddpg import DDPGAgent, TD3Agent

FACTORIES = {"sac": SACAgent, "ddpg": DDPGAgent, "td3": TD3Agent}


def episode_stat(ep_s, ep_a, ep_r, ret, reach, H, clip=1.0):
    v = np.array([
        ep_s[:, 0].mean(), ep_s[:, 1].mean(), ep_s[:, 2].mean(), ep_s[:, 3].mean(),
        ep_a[:, 0].mean(), ep_a[:, 1].mean(),
        ep_r.mean(), ret, float(reach), len(ep_r) / H,
    ], dtype=np.float32)
    n = np.linalg.norm(v)
    return v * (clip / n) if n > clip else v


def train_agent(cfg, env, seed, device, sectors=None, inverse_unlearn=False,
                collect_stats=False, log_every=10_000, algo="sac"):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    agent = FACTORIES[algo](cfg, device)
    buf = ReplayBuffer(cfg.buffer_cap, cfg.state_dim, cfg.action_dim, cfg.n_sectors)
    allowed = list(range(cfg.n_sectors)) if sectors is None else list(sectors)
    ep_stats = []

    if inverse_unlearn:
        tag = f"non-transfer-LFS[{algo}]"
    elif sectors is None:
        tag = f"original(all-sectors)[{algo}]"
    else:
        tag = f"oracle(retain-only)[{algo}]"

    s = env.reset(rng, sector=int(rng.choice(allowed)))
    ep_s, ep_a, ep_r = [], [], []
    ep_ret, ep_reach = 0.0, False
    ep_count = 0
    recent_returns = []

    for step in range(cfg.train_steps):
        if log_every and step % log_every == 0:
            m = float(np.mean(recent_returns[-50:])) if recent_returns else float("nan")
            print(f"  [seed {seed} | {tag}] step {step}/{cfg.train_steps} "
                  f"| episodes {ep_count} | last-50 mean return {m:.2f}", flush=True)

        a = (agent.act(s) if step >= cfg.warmup
             else rng.uniform(-1, 1, cfg.action_dim).astype(np.float32))
        s2, r, done, info = env.step(a)
        buf.add(s, a, r, s2, float(done), info["sector"])
        ep_s.append(s); ep_a.append(a); ep_r.append(r)
        ep_ret += r
        ep_reach = ep_reach or info["reach"]
        s = s2

        if done:
            ep_count += 1
            recent_returns.append(ep_ret)
            if collect_stats:
                stat = episode_stat(np.array(ep_s), np.array(ep_a),
                                    np.array(ep_r), ep_ret, ep_reach,
                                    cfg.episode_len)
                ep_stats.append((stat, env.episode_sector))
            s = env.reset(rng, sector=int(rng.choice(allowed)))
            ep_s, ep_a, ep_r = [], [], []
            ep_ret, ep_reach = 0.0, False

        if step >= cfg.warmup:
            if inverse_unlearn:
                b = _sample_tagged(buf, cfg.batch, device, rng, cfg)
                inv = (b["tag"] == cfg.unlearn_sector).float().unsqueeze(1)
                b["r"] = b["r"] * (1.0 - 2.0 * inv)
            else:
                b = buf.sample(cfg.batch, device, rng=rng)
            agent.update(b)

    if log_every:
        m = float(np.mean(recent_returns[-50:])) if recent_returns else float("nan")
        print(f"  [seed {seed} | {tag}] finished {cfg.train_steps} steps "
              f"| episodes {ep_count} | last-50 mean return {m:.2f}", flush=True)
    return agent, buf, ep_stats


def _sample_tagged(buf, batch, device, rng, cfg):
    idx = rng.integers(0, buf.size, batch)
    t = lambda x: torch.as_tensor(x[idx], device=device)
    return {"s": t(buf.s), "a": t(buf.a), "r": t(buf.r), "s2": t(buf.s2),
            "d": t(buf.d), "tag": t(buf.sector)}
