from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Config:
    # ---- environment ----
    n_sectors: int = 4
    unlearn_sector: int = 0
    episode_len: int = 100
    state_dim: int = 6
    action_dim: int = 2

    # ---- agents ----
    algos: Tuple[str, ...] = ("sac", "ddpg", "td3")
    hidden: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 5e-3
    alpha: float = 0.2
    explore_sigma: float = 0.2      # DDPG/TD3 behaviour noise
    policy_noise: float = 0.2       # TD3 target-policy smoothing
    noise_clip: float = 0.5
    policy_delay: int = 2
    buffer_cap: int = 300_000
    batch: int = 256
    warmup: int = 1_000
    train_steps: int = 60_000

    # ---- unlearning / ablations ----
    unlearn_steps: int = 6_000
    lam_kl: float = 1.0
    w_retain: float = 1.0
    q_min: float = 0.0
    poison_noise: float = 0.1
    ablations: Tuple[str, ...] = ("full", "no_poison", "no_anchor",
                                  "no_rewind", "no_hinge")
    lambda_grid: Tuple[float, ...] = (0.0, 0.1, 1.0, 10.0)
    sweep_steps: int = 1_500

    # ---- TV-stability tree ----
    rho: float = 0.1
    rho_grid: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.4)
    stat_dim: int = 10
    coupling_leaves: int = 32

    # ---- evaluation ----
    eval_episodes: int = 40
    adv_episodes: int = 60
    stop_eval_episodes: int = 10
    stop_tol: float = 1.0
    eval_every: int = 100

    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    device: str = "auto"
    out_dir: str = "results"
