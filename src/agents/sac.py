"""Soft Actor-Critic (continuous actions) + tagged replay buffer."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def kl_gauss(m1, l1, m2, l2):
    """KL( N(m1,s1) || N(m2,s2) ), summed over action dims, mean over batch."""
    s1, s2 = l1.exp(), l2.exp()
    return (l2 - l1 + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5).sum(-1).mean()


class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def gauss_params(self, s):
        h = self.trunk(s)
        return self.mean(h), self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)

    def forward(self, s):
        return self.sample(s)

    def sample(self, s, deterministic=False):
        mean, log_std = self.gauss_params(s)
        if deterministic:
            return torch.tanh(mean), None
        dist = Normal(mean, log_std.exp())
        x = dist.rsample()
        a = torch.tanh(x)
        logp = (dist.log_prob(x) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return a, logp


class TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1),
        )

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_only(self, s, a):
        return self.q1(torch.cat([s, a], dim=-1))


class ReplayBuffer:
    def __init__(self, cap, sdim, adim, n_sectors):
        self.cap = cap
        self.s = np.zeros((cap, sdim), np.float32)
        self.a = np.zeros((cap, adim), np.float32)
        self.r = np.zeros((cap, 1), np.float32)
        self.s2 = np.zeros((cap, sdim), np.float32)
        self.d = np.zeros((cap, 1), np.float32)
        self.sector = np.zeros(cap, np.int64)
        self.ptr = 0
        self.size = 0
        self.idx_by_sector = {k: [] for k in range(n_sectors)}

    def add(self, s, a, r, s2, d, k):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s2[self.ptr] = s2
        self.d[self.ptr] = d
        self.sector[self.ptr] = k
        if len(self.idx_by_sector[k]) < self.cap:
            self.idx_by_sector[k].append(self.ptr)
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch, device, sectors=None, rng=None):
        if sectors is None:
            idx = rng.integers(0, self.size, batch)
        else:
            pool = np.concatenate([self.idx_by_sector[k] for k in sectors])
            idx = pool[rng.integers(0, len(pool), batch)]
        t = lambda x: torch.as_tensor(x[idx], device=device)
        return {"s": t(self.s), "a": t(self.a), "r": t(self.r),
                "s2": t(self.s2), "d": t(self.d)}


class SACAgent:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.dev = device
        self.actor = GaussianActor(cfg.state_dim, cfg.action_dim, cfg.hidden).to(device)
        self.critic = TwinCritic(cfg.state_dim, cfg.action_dim, cfg.hidden).to(device)
        self.critic_t = TwinCritic(cfg.state_dim, cfg.action_dim, cfg.hidden).to(device)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def act(self, s_np, deterministic=False):
        s = torch.as_tensor(s_np, device=self.dev).unsqueeze(0)
        a, _ = self.actor.sample(s, deterministic=deterministic)
        return a.squeeze(0).cpu().numpy()

    def _td_targets(self, b):
        with torch.no_grad():
            a2, logp2 = self.actor.sample(b["s2"])
            q1t, q2t = self.critic_t(b["s2"], a2)
            qt = torch.min(q1t, q2t) - self.cfg.alpha * logp2
            y = b["r"] + (1.0 - b["d"]) * self.cfg.gamma * qt
        return y

    def update(self, b):
        y = self._td_targets(b)
        q1, q2 = self.critic(b["s"], b["a"])
        c_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_c.zero_grad()
        c_loss.backward()
        self.opt_c.step()

        a, logp = self.actor.sample(b["s"])
        q1 = self.critic.q1_only(b["s"], a)
        a_loss = (self.cfg.alpha * logp - q1).mean()
        self.opt_a.zero_grad()
        a_loss.backward()
        self.opt_a.step()

        for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
            pt.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
        return {"critic": float(c_loss), "actor": float(a_loss)}
