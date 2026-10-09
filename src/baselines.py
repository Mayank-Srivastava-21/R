"""Baselines: Oracle retrain (LFS) and Non-transferable LFS, per algorithm."""
from .agents.trainer import train_agent


def oracle_lfs(cfg, env, seed, device, algo="sac"):
    retain = [k for k in range(cfg.n_sectors) if k != cfg.unlearn_sector]
    agent, buf, _ = train_agent(cfg, env, seed, device, sectors=retain, algo=algo)
    return agent


def non_transfer_lfs(cfg, env, seed, device, algo="sac"):
    agent, buf, _ = train_agent(cfg, env, seed, device, inverse_unlearn=True, algo=algo)
    return agent
