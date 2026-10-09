"""Certified decremental unlearning with explicit ablation flags.

Actor objective  =  w_retain * (-Q_retain)            [utility term]
                 +  hinge(Q_unlearn - Q_min)          [decremental term]
                 +  lam * Anchor(pi_orig || pi_theta) [KL (stochastic) or
                                                       action-MSE (deterministic)]
Critic objective =  TD(retain) + TD(poison)

Flags per mode (ablations):
  full      : poison+anchor+hinge+rewind+retain-term   (our method)
  no_poison : -poisoning
  no_anchor : -anchor (KL / action-MSE)
  no_rewind : -reference-matched early stopping
  no_hinge  : -decremental hinge (poisoning+anchor only)
  cf        : hinge only (catastrophic-forgetting baseline)
"""
import copy
import time

import torch
import torch.nn.functional as F

from ..agents.sac import kl_gauss
from .poisoning import sample_poison

MODE_DEFAULTS = {
    "full":          dict(poison=True,  anchor=True,  hinge=True,  rewind=True,  retain_term=True),
    "ours":          dict(poison=True,  anchor=True,  hinge=True,  rewind=True,  retain_term=True),
    "no_poison":     dict(poison=False, anchor=True,  hinge=True,  rewind=True,  retain_term=True),
    "ours_nopoison": dict(poison=False, anchor=True,  hinge=True,  rewind=True,  retain_term=True),
    "no_anchor":     dict(poison=True,  anchor=False, hinge=True,  rewind=True,  retain_term=True),
    "no_rewind":     dict(poison=True,  anchor=True,  hinge=True,  rewind=False, retain_term=True),
    "no_hinge":      dict(poison=True,  anchor=True,  hinge=False, rewind=True,  retain_term=True),
    "cf":            dict(poison=False, anchor=False, hinge=True,  rewind=False, retain_term=False),
}


def _actions(agent, s):
    if hasattr(agent.actor, "sample"):          # stochastic (SAC)
        a, _ = agent.actor.sample(s)
    else:                                       # deterministic (DDPG/TD3)
        a = agent.actor(s)
    return a


def _anchor(agent, orig, s):
    if hasattr(agent.actor, "gauss_params"):    # KL between Gaussians
        with torch.no_grad():
            m_o, l_o = orig.gauss_params(s)
        m_c, l_c = agent.actor.gauss_params(s)
        return kl_gauss(m_o, l_o, m_c, l_c)
    with torch.no_grad():                       # action-space MSE
        a_o = orig(s)
    return ((agent.actor(s) - a_o) ** 2).sum(-1).mean()


def run_unlearn(agent, orig_actor, cfg, buf, poison, device, rng,
                mode="full", stop_fn=None, stop_ref=None, retain_fn=None,
                stop_tol=1.0, eval_every=100, lam_kl=None, max_steps=None):
    flags = dict(MODE_DEFAULTS[mode])
    lam = cfg.lam_kl if lam_kl is None else lam_kl
    agent = copy.deepcopy(agent)
    orig = copy.deepcopy(orig_actor).to(device).eval()
    for p in orig.parameters():
        p.requires_grad_(False)
    opt_a = torch.optim.Adam(agent.actor.parameters(), lr=cfg.lr)
    opt_c = torch.optim.Adam(agent.critic.parameters(), lr=cfg.lr)
    retain = [k for k in range(cfg.n_sectors) if k != cfg.unlearn_sector]
    limit = cfg.unlearn_steps if max_steps is None else max_steps

    history, stopped = [], False
    t0 = time.perf_counter()
    for step in range(limit):
        b_r = buf.sample(cfg.batch, device, sectors=retain, rng=rng)
        b_u = buf.sample(cfg.batch, device, sectors=[cfg.unlearn_sector], rng=rng)

        # ---------------- critic ----------------
        y_r = agent._td_targets(b_r)
        q1, q2 = agent.critic(b_r["s"], b_r["a"])
        c_loss = F.mse_loss(q1, y_r) + F.mse_loss(q2, y_r)
        if flags["poison"]:
            b_p = sample_poison(poison, cfg.batch, device, rng)
            y_p = agent._td_targets(b_p)
            q1p, q2p = agent.critic(b_p["s"], b_p["a"])
            c_loss = c_loss + F.mse_loss(q1p, y_p) + F.mse_loss(q2p, y_p)
        opt_c.zero_grad(); c_loss.backward(); opt_c.step()

        # ---------------- actor ----------------
        a_r = _actions(agent, b_r["s"])
        q_r = agent.critic.q1_only(b_r["s"], a_r)
        a_u = _actions(agent, b_u["s"])
        q_u = agent.critic.q1_only(b_u["s"], a_u)
        terms = []
        if flags["retain_term"]:
            terms.append(cfg.w_retain * (-q_r.mean()))
        if flags["hinge"]:
            terms.append(F.relu(q_u - cfg.q_min).mean())
        if flags["anchor"]:
            terms.append(lam * _anchor(agent, orig, b_r["s"]))
        a_loss = sum(terms) if terms else (q_u * 0.0).sum()
        opt_a.zero_grad(); a_loss.backward(); opt_a.step()

        for p, pt in zip(agent.critic.parameters(), agent.critic_t.parameters()):
            pt.data.mul_(1 - cfg.tau).add_(cfg.tau * p.data)
        if hasattr(agent, "actor_t"):
            for p, pt in zip(agent.actor.parameters(), agent.actor_t.parameters()):
                pt.data.mul_(1 - cfg.tau).add_(cfg.tau * p.data)

        # ---------------- history + rewinding ----------------
        if (step + 1) % eval_every == 0:
            rec = {"step": step + 1}
            if stop_fn is not None:
                rec["revoked"] = stop_fn(agent)
            if retain_fn is not None:
                rec["retained"] = retain_fn(agent)
            history.append(rec)
            if flags["rewind"] and stop_fn is not None and stop_ref is not None:
                if rec.get("revoked", 1e9) <= stop_ref + stop_tol:
                    stopped = True
                    break

    elapsed = time.perf_counter() - t0
    info = {"mode": mode, "stopped_early": stopped, "steps": step + 1,
            "history": history, "flags": flags}
    agent.stop_info = info
    return agent, elapsed, info
