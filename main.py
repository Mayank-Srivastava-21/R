"""End-to-end orchestrator: multi-algorithm (SAC/DDPG/TD3) certified
reinforcement unlearning with ablations, sweeps, ROC, and CSV export.

CHANGES IN THIS REVISION
  1. aggregate(): mean ROC curves are stored as plain Python lists
     (fpr.tolist(), tpr.tolist()) so results are always JSON-serializable.
  2. (utils.save_json now also has a numpy/torch fallback serializer.)

Usage:  python -u main.py --seeds 0 1 2
"""
import argparse
import copy
import math
import time
from dataclasses import replace

import numpy as np
import torch

from config import Config
from src.utils import set_seed, get_device, ensure_dir, save_json
from src.envs.sector_nav import SectorNavEnv
from src.agents.trainer import train_agent
from src.baselines import oracle_lfs, non_transfer_lfs
from src.unlearning.poisoning import build_poison_buffer
from src.unlearning.decremental import run_unlearn
from src.unlearning.binary_tree import BinaryTree
from src.attacks.env_inference import (rollout_features, train_attacker,
                                       pair_accuracy, pair_auc, pair_roc)
from src.eval.metrics import evaluate
from src import visualization as viz

STOP_TOL = 1.0


def _safe(fn, *args, **kwargs):
    """Never let a plotting bug kill a multi-hour run."""
    try:
        fn(*args, **kwargs)
    except Exception as e:
        print(f"[warn] {fn.__name__} failed: {type(e).__name__}: {e}")


def coupling_demo(cfg, ep_stats, rng, rho):
    T = max(len(ep_stats), 2)
    revoked = [i for i, (x, k) in enumerate(ep_stats)
               if k == cfg.unlearn_sector][: cfg.coupling_leaves]

    def poisoned(x):
        xp = x.copy(); xp[4:8] = -xp[4:8]; xp[8] = 0.0
        return xp

    deltas = [np.linalg.norm(ep_stats[i][0] - poisoned(ep_stats[i][0])) for i in revoked]
    B = float(max(deltas)) if deltas else 1.0
    cap = max(64, 2 ** int(math.ceil(math.log2(max(2, T)))))
    L = float(int(math.log2(cap)) + 1)
    sigma = B * math.sqrt(L) / (2.0 * rho)
    bound = rho * math.sqrt(L)
    tree = BinaryTree(cap=cap, dim=cfg.stat_dim, sigma=sigma, rng=rng)
    for i, (x, k) in enumerate(ep_stats):
        tree.tree_update(x, i)
    retrains = sum(int(tree.unlearn(i, poisoned(ep_stats[i][0]))[0]) for i in revoked)
    return {"rho": rho, "empirical": retrains / max(1, len(revoked)),
            "bound": bound, "sigma": sigma, "B": B, "T": T, "L": L,
            "n_revoked_leaves": len(revoked)}


def run_seed(cfg, seed, device, algo):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    env = SectorNavEnv(n_sectors=cfg.n_sectors, episode_len=cfg.episode_len)
    retain = [k for k in range(cfg.n_sectors) if k != cfg.unlearn_sector]

    t0 = time.perf_counter()
    agent_orig, buf, ep_stats = train_agent(cfg, env, seed, device,
                                            collect_stats=True, algo=algo)
    t_train = time.perf_counter() - t0

    poison = build_poison_buffer(buf, cfg, rng)
    with torch.no_grad():
        orig_actor = copy.deepcopy(agent_orig.actor).eval()

    probe, _, _ = train_agent(replace(cfg, train_steps=max(2000, cfg.train_steps // 4)),
                              env, seed + 101, device, sectors=retain,
                              log_every=0, algo=algo)
    r_ref, _ = evaluate(probe, env, cfg.unlearn_sector, cfg.eval_episodes, seed + 102, cfg)
    stop_fn = lambda ag: evaluate(ag, env, cfg.unlearn_sector,
                                  cfg.stop_eval_episodes, seed + 103, cfg)[0]
    retain_fn = lambda ag: float(np.mean(
        [evaluate(ag, env, k, 5, seed + 104, cfg)[0] for k in retain]))

    # ---- ablation family ----
    ab = {}
    for name in cfg.ablations:
        ag, t, info = run_unlearn(agent_orig, orig_actor, cfg, buf, poison, device, rng,
                                  mode=name, stop_fn=stop_fn, stop_ref=r_ref,
                                  retain_fn=retain_fn, stop_tol=STOP_TOL,
                                  eval_every=cfg.eval_every)
        ab[name] = (ag, t, info)
    ag_cf, t_cf, _ = run_unlearn(agent_orig, orig_actor, cfg, buf, poison, device,
                                 rng, mode="cf")

    t0 = time.perf_counter()
    agent_oracle = oracle_lfs(cfg, env, seed, device, algo)
    t_retrain = time.perf_counter() - t0
    agent_nt = non_transfer_lfs(cfg, env, seed, device, algo)

    coup = coupling_demo(cfg, ep_stats, rng, cfg.rho)
    rho_rows = [coupling_demo(cfg, ep_stats, rng, r) for r in cfg.rho_grid]

    policies = {"original": agent_orig, "cf": ag_cf,
                "oracle_lfs": agent_oracle, "nontransfer_lfs": agent_nt}
    policies.update({k: v[0] for k, v in ab.items()})

    ret_sec = {n: {f"S{k}": evaluate(p, env, k, cfg.eval_episodes, seed, cfg)[0]
                   for k in range(cfg.n_sectors)} for n, p in policies.items()}
    ret_random = {f"S{k}": evaluate(None, env, k, cfg.eval_episodes, seed, cfg)[0]
                  for k in range(cfg.n_sectors)}

    # ---- attacks ----
    f_seen = rollout_features(agent_orig, env, cfg.unlearn_sector, cfg.adv_episodes, seed, cfg)
    f_unseen = rollout_features(agent_oracle, env, cfg.unlearn_sector, cfg.adv_episodes, seed, cfg)
    att = train_attacker(f_seen, f_unseen, device, seed=seed)
    acc, auc, roc = {}, {}, {}
    acc["sanity_seen_vs_unseen"] = pair_accuracy(att, f_seen, f_unseen, device)
    auc["sanity_seen_vs_unseen"] = pair_auc(att, f_seen, f_unseen, device)
    for name in list(ab.keys()) + ["cf", "nontransfer_lfs"]:
        f_u = rollout_features(policies[name], env, cfg.unlearn_sector,
                               cfg.adv_episodes, seed, cfg)
        acc[f"{name}_vs_unseen"] = pair_accuracy(att, f_u, f_unseen, device)
        auc[f"{name}_vs_unseen"] = pair_auc(att, f_u, f_unseen, device)
        acc[f"{name}_vs_seen"] = pair_accuracy(att, f_u, f_seen, device)
        if name == "full":
            fpr, tpr = pair_roc(att, f_u, f_unseen, device)
            roc["full_vs_unseen"] = (fpr, tpr, auc["full_vs_unseen"])
            fpr2, tpr2 = pair_roc(att, f_seen, f_unseen, device)
            roc["sanity_seen_vs_unseen"] = (fpr2, tpr2, auc["sanity_seen_vs_unseen"])

    # ---- lambda sweep (anchor strength) ----
    lam_rows = []
    for lam in cfg.lambda_grid:
        ag_l, _, _ = run_unlearn(agent_orig, orig_actor, cfg, buf, poison, device, rng,
                                 mode="full", lam_kl=lam, max_steps=cfg.sweep_steps,
                                 stop_fn=stop_fn, stop_ref=r_ref, stop_tol=STOP_TOL,
                                 eval_every=cfg.eval_every)
        r_ret = float(np.mean([evaluate(ag_l, env, k, cfg.eval_episodes, seed, cfg)[0]
                               for k in retain]))
        r_rev = evaluate(ag_l, env, cfg.unlearn_sector, cfg.eval_episodes, seed, cfg)[0]
        f_l = rollout_features(ag_l, env, cfg.unlearn_sector, cfg.adv_episodes, seed, cfg)
        lam_rows.append({"lambda": lam, "retained": r_ret, "revoked": r_rev,
                         "infer_acc": pair_accuracy(att, f_l, f_unseen, device)})

    return {"seed": seed, "algo": algo, "returns": ret_sec, "random_returns": ret_random,
            "attack_acc": acc, "attack_auc": auc,
            "roc": {k: [v[0].tolist(), v[1].tolist(), float(v[2])] for k, v in roc.items()},
            "times": {"train": t_train, "full": ab["full"][1], "cf": t_cf,
                      "retrain_oracle": t_retrain},
            "ab_times": {k: v[1] for k, v in ab.items()},
            "histories": {k: v[2]["history"] for k, v in ab.items()},
            "coupling": coup, "rho_rows": rho_rows, "lam_rows": lam_rows,
            "r_ref_probe": float(r_ref),
            "rewind": {k: bool(v[2]["stopped_early"]) for k, v in ab.items()}}


def aggregate(per_seed, cfg):
    methods = list(per_seed[0]["returns"].keys())
    sectors = [f"S{k}" for k in range(cfg.n_sectors)]
    per_sector = {m: {s: float(np.mean([p["returns"][m][s] for p in per_seed]))
                      for s in sectors} for m in methods}
    util = {m: float(np.mean([np.mean([p["returns"][m][s] for s in sectors[1:]])
                              for p in per_seed])) for m in methods}
    rev = {m: float(np.mean([p["returns"][m]["S0"] for p in per_seed])) for m in methods}
    accu = {m: float(np.mean([p["attack_acc"][f"{m}_vs_unseen"] for p in per_seed]))
            for m in methods if f"{m}_vs_unseen" in per_seed[0]["attack_acc"]}
    pareto = {m: {"acc": accu[m], "utility": util[m]} for m in accu}
    pareto["oracle_lfs"] = {"acc": 0.5, "utility": util["oracle_lfs"]}
    cost = {"ours (full)": float(np.mean([p["times"]["full"] for p in per_seed])),
            "cf fine-tune": float(np.mean([p["times"]["cf"] for p in per_seed])),
            "retrain-from-scratch (oracle)": float(np.mean([p["times"]["retrain_oracle"] for p in per_seed]))}
    coupling = {"empirical": float(np.mean([p["coupling"]["empirical"] for p in per_seed])),
                "bound": float(np.mean([p["coupling"]["bound"] for p in per_seed]))}
    ab_rows = []
    for abl in cfg.ablations:
        ab_rows.append({
            "ablation": abl,
            "retained": util[abl], "revoked": rev[abl],
            "infer_acc": accu.get(abl, float("nan")),
            "seconds": float(np.mean([p["ab_times"][abl] for p in per_seed]))})
    rho_rows = [{"rho": r,
                 "empirical": float(np.mean([p["rho_rows"][i]["empirical"] for p in per_seed])),
                 "bound": float(np.mean([p["rho_rows"][i]["bound"] for p in per_seed]))}
                for i, r in enumerate(cfg.rho_grid)]
    lam_rows = [{"lambda": v,
                 "retained": float(np.mean([p["lam_rows"][i]["retained"] for p in per_seed])),
                 "revoked": float(np.mean([p["lam_rows"][i]["revoked"] for p in per_seed])),
                 "infer_acc": float(np.mean([p["lam_rows"][i]["infer_acc"] for p in per_seed]))}
                for i, v in enumerate(cfg.lambda_grid)]
    roc = {}
    for key in per_seed[0]["roc"]:
        fpr = np.mean([np.array(p["roc"][key][0]) for p in per_seed], axis=0)
        tpr = np.mean([np.array(p["roc"][key][1]) for p in per_seed], axis=0)
        # FIX: store as plain lists so the aggregate is JSON-serializable
        roc[key] = (fpr.tolist(), tpr.tolist(),
                    float(np.mean([p["roc"][key][2] for p in per_seed])))
    histories = {k: [h for p in per_seed for h in p["histories"][k]]
                 for k in per_seed[0]["histories"]}
    headline = {
        "retained_utility_orig": util["original"], "retained_utility_full": util["full"],
        "utility_loss_pct": 100.0 * (util["original"] - util["full"]) / max(1e-9, abs(util["original"])),
        "revoked_full": rev["full"], "revoked_oracle": rev["oracle_lfs"],
        "inference_acc_full_vs_unseen": accu["full"],
        "inference_auc_full_vs_unseen": float(np.mean([p["attack_auc"]["full_vs_unseen"] for p in per_seed])),
        "unlearn_time_full": cost["ours (full)"], "retrain_time_oracle": cost["retrain-from-scratch (oracle)"],
        "speedup": cost["retrain-from-scratch (oracle)"] / max(1e-9, cost["ours (full)"]),
        "coupling_empirical": coupling["empirical"], "coupling_bound": coupling["bound"],
        "rewind_frac": float(np.mean([float(p["rewind"]["full"]) for p in per_seed]))}
    return {"per_sector_returns": per_sector, "pareto": pareto, "cost": cost,
            "coupling": coupling, "ablations": ab_rows, "rho_rows": rho_rows,
            "lam_rows": lam_rows, "roc": roc, "histories": histories,
            "headline": headline,
            "r_ref": float(np.mean([p["r_ref_probe"] for p in per_seed]))}


def emit(algo, agg, cfg, out):
    f = f"{out}/figs/{algo}_"
    c = f"{out}/csv/{algo}_"
    _safe(viz.plot_sector_returns, agg, f + "sector_returns.png", c + "sector_returns.csv")
    _safe(viz.plot_pareto, agg, f + "pareto.png", c + "pareto.csv")
    _safe(viz.plot_cost, agg, f + "cost.png", c + "cost.csv")
    _safe(viz.plot_coupling, agg, f + "coupling.png", c + "coupling.csv")
    _safe(viz.plot_dynamics, agg["histories"], agg["r_ref"], cfg.stop_tol,
          f + "unlearn_dynamics.png", c + "unlearn_dynamics.csv")
    _safe(viz.plot_rho_sweep, agg["rho_rows"], f + "rho_sweep.png", c + "rho_sweep.csv")
    _safe(viz.plot_lambda_sweep, agg["lam_rows"], f + "lambda_sweep.png", c + "lambda_sweep.csv")
    _safe(viz.plot_roc, agg["roc"], f + "roc.png", c + "roc.csv")
    _safe(viz.plot_ablations, agg["ablations"], f + "ablations.png", c + "ablations.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--algos", type=str, nargs="+", default=None)
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    cfg = Config(seeds=args.seeds)
    if args.algos:
        cfg.algos = tuple(args.algos)
    if args.steps:
        cfg.train_steps = args.steps
        cfg.unlearn_steps = max(1000, args.steps // 10)
        cfg.sweep_steps = max(300, args.steps // 40)
    device = get_device(cfg.device)
    for d in (args.out, args.out + "/figs", args.out + "/csv"):
        ensure_dir(d)
    print(f"[device] {device} | algos={cfg.algos}")

    results = {}
    algo_rows = []
    for algo in cfg.algos:
        per_seed = [run_seed(cfg, s, device, algo) for s in cfg.seeds]
        agg = aggregate(per_seed, cfg)
        results[algo] = {"per_seed": per_seed, "aggregate": agg}
        # ---- SAVE FIRST, plot second: never lose compute to a downstream bug ----
        save_json(results[algo], f"{args.out}/results_{algo}.json")
        emit(algo, agg, cfg, args.out)
        h = agg["headline"]
        algo_rows.append({"algo": algo, "retained": h["retained_utility_full"],
                          "revoked": h["revoked_full"],
                          "infer_acc": h["inference_acc_full_vs_unseen"],
                          "speedup": h["speedup"]})
        print("\n=== [%s] %s\n" % (algo, " | ".join(f"{k}={v:.4f}" for k, v in h.items())))

    _safe(viz.plot_algos, algo_rows, f"{args.out}/figs/algo_comparison.png",
          f"{args.out}/csv/global_algo_comparison.csv")
    save_json(results, f"{args.out}/results.json")
    print(f"Figures + CSVs + JSON saved under {args.out}/")


if __name__ == "__main__":
    main()
