"""Deterministic policy-gradient agents (DDPG, TD3) with the same public API
as SACAgent so the unlearning stack is algorithm-agnostic."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sac import TwinCritic


class DeterministicActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, action_dim)

    def forward(self, s):
        return torch.tanh(self.head(self.trunk(s)))


class DDPGAgent:
    variant = "ddpg"

    def __init__(self, cfg, device):
        self.cfg, self.dev = cfg, device
        self.actor = DeterministicActor(cfg.state_dim, cfg.action_dim, cfg.hidden).to(device)
        self.actor_t = DeterministicActor(cfg.state_dim, cfg.action_dim, cfg.hidden).to(device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic = TwinCritic(cfg.state_dim, cfg.action_dim, cfg.hidden).to(device)
        self.critic_t = TwinCritic(cfg.state_dim, cfg.action_dim, cfg.hidden).to(device)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.it = 0

    def params(self):
        return list(self.actor.parameters()) + list(self.critic.parameters())

    @torch.no_grad()
    def act(self, s_np, deterministic=False):
        s = torch.as_tensor(s_np, device=self.dev, dtype=torch.float32).unsqueeze(0)
        a = self.actor(s).squeeze(0).cpu().numpy()
        if not deterministic:
            a = a + self.cfg.explore_sigma * np.random.randn(*a.shape)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def _td_targets(self, b):
        with torch.no_grad():
            a2 = self.actor_t(b["s2"])
            qt = self.critic_t.q1_only(b["s2"], a2)
            y = b["r"] + (1.0 - b["d"]) * self.cfg.gamma * qt
        return y

    def _soft(self):
        for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()):
            pt.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
        for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
            pt.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

    def update(self, b):
        y = self._td_targets(b)
        q1, q2 = self.critic(b["s"], b["a"])
        c_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_c.zero_grad(); c_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.opt_c.step()

        a = self.actor(b["s"])
        a_loss = -self.critic.q1_only(b["s"], a).mean()
        self.opt_a.zero_grad(); a_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.opt_a.step()
        self._soft()
        return {"critic": float(c_loss), "actor": float(a_loss)}


class TD3Agent(DDPGAgent):
    variant = "td3"

    def _td_targets(self, b):
        with torch.no_grad():
            a2 = self.actor_t(b["s2"])
            eps = (torch.randn_like(a2) * self.cfg.policy_noise).clamp(
                -self.cfg.noise_clip, self.cfg.noise_clip)
            a2 = (a2 + eps).clamp(-1.0, 1.0)
            q1t, q2t = self.critic_t(b["s2"], a2)
            qt = torch.min(q1t, q2t)
            y = b["r"] + (1.0 - b["d"]) * self.cfg.gamma * qt
        return y

    def update(self, b):
        y = self._td_targets(b)
        q1, q2 = self.critic(b["s"], b["a"])
        c_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_c.zero_grad(); c_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.opt_c.step()

        self.it += 1
        if self.it % self.cfg.policy_delay == 0:
            a = self.actor(b["s"])
            a_loss = -self.critic.q1_only(b["s"], a).mean()
            self.opt_a.zero_grad(); a_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.opt_a.step()
            self._soft()
        return {"critic": float(c_loss)}
