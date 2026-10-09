import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ---------------- core four (with CSV) ----------------
def plot_sector_returns(res, path, csv_path=None):
    methods = list(res["per_sector_returns"].keys())
    sectors = sorted(res["per_sector_returns"][methods[0]].keys())
    x = np.arange(len(sectors)); w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, m in enumerate(methods):
        ax.bar(x + i * w, [res["per_sector_returns"][m][s] for s in sectors], w, label=m)
    ax.axvline(0.5, color="red", ls="--", lw=1, label="revoked sector boundary")
    ax.set_xticks(x + 0.4); ax.set_xticklabels(sectors)
    ax.set_ylabel("mean return"); ax.set_title("Per-sector utility after unlearning")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    if csv_path:
        save_csv(csv_path, ["method"] + sectors,
                 [[m] + [res["per_sector_returns"][m][s] for s in sectors] for m in methods])


def plot_pareto(res, path, csv_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    for m, d in res["pareto"].items():
        ax.scatter(d["acc"], d["utility"], s=90, label=m)
    ax.axvline(0.5, color="gray", ls=":", label="random guessing")
    ax.set_xlabel("environment-inference accuracy (unlearned vs never-seen)")
    ax.set_ylabel("retained utility (mean return)")
    ax.set_title("Privacy-Utility Pareto front"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    if csv_path:
        save_csv(csv_path, ["method", "infer_acc", "retained_utility"],
                 [[m, d["acc"], d["utility"]] for m, d in res["pareto"].items()])


def plot_cost(res, path, csv_path=None):
    items = res["cost"]; names = list(items.keys())
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(names, [items[k] for k in names], color=["#8fd0a8", "#3fb6c4", "#2b7bb8"])
    ax.set_xscale("log"); ax.set_xlabel("wall-clock seconds (log)")
    ax.set_title("Unlearning cost vs retraining (Theorem 3)")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    if csv_path:
        save_csv(csv_path, ["method", "seconds"], [[k, items[k]] for k in names])


def plot_coupling(res, path, csv_path=None):
    c = res["coupling"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["empirical retrain rate", r"bound $\rho\sqrt{\ln T}$"],
           [c["empirical"], c["bound"]], color=["#2b7bb8", "#e34a33"])
    ax.set_ylabel("P(retrain)"); ax.set_title("Maximal-coupling cost certification")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    if csv_path:
        save_csv(csv_path, ["quantity", "value"],
                 [["empirical", c["empirical"]], ["bound", c["bound"]]])


# ---------------- new figures ----------------
def plot_dynamics(histories, r_ref, tol, path, csv_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    rows = []
    for m, hist in histories.items():
        st = [h["step"] for h in hist]
        rv = [h.get("revoked", np.nan) for h in hist]
        rt = [h.get("retained", np.nan) for h in hist]
        axes[0].plot(st, rv, marker="o", ms=3, label=m)
        axes[1].plot(st, rt, marker="o", ms=3, label=m)
        rows += [[m, s, v, w] for s, v, w in zip(st, rv, rt)]
    axes[0].axhline(r_ref, color="k", ls="--", lw=1, label="never-seen ref r_ref")
    axes[0].axhspan(r_ref - tol, r_ref + tol, color="gray", alpha=0.2)
    axes[0].set_xlabel("unlearning step"); axes[0].set_ylabel("revoked-sector return")
    axes[0].set_title("Forgetting dynamics + rewinding band")
    axes[1].set_xlabel("unlearning step"); axes[1].set_ylabel("retained return")
    axes[1].set_title("Utility during unlearning")
    for a in axes:
        a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    if csv_path:
        save_csv(csv_path, ["method", "step", "revoked_return", "retained_return"], rows)


def plot_rho_sweep(rows, path, csv_path=None):
    rh = [r["rho"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(rh, [r["empirical"] for r in rows], "o-", label="empirical retrain rate")
    ax.plot(rh, [r["bound"] for r in rows], "s--", label=r"bound $\rho\sqrt{L}$")
    ax.set_xlabel(r"stability parameter $\rho$"); ax.set_ylabel("P(retrain)")
    ax.set_title(r"Certification holds across $\rho$ (Thm 3)")   # FIX: raw string
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    if csv_path:
        save_csv(csv_path, ["rho", "empirical", "bound"],
                 [[r["rho"], r["empirical"], r["bound"]] for r in rows])


def plot_lambda_sweep(rows, path, csv_path=None):
    lm = [r["lambda"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(lm, [r["retained"] for r in rows], "o-", label="retained utility")
    axes[0].plot(lm, [r["revoked"] for r in rows], "s-", label="revoked return")
    axes[0].set_xscale("symlog", linthresh=0.01)
    axes[1].plot(lm, [r["infer_acc"] for r in rows], "o-", color="crimson")
    axes[1].axhline(0.5, color="gray", ls=":")
    axes[1].set_ylim(0, 1); axes[1].set_xscale("symlog", linthresh=0.01)
    axes[0].set_xlabel(r"$\lambda$ (anchor strength)"); axes[0].set_ylabel("return")
    axes[1].set_xlabel(r"$\lambda$"); axes[1].set_ylabel("infer acc vs never-seen")
    axes[0].set_title("Utility/forgetting vs anchor"); axes[0].legend(fontsize=8)
    axes[1].set_title("Privacy vs anchor")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    if csv_path:
        save_csv(csv_path, ["lambda", "retained", "revoked", "infer_acc"],
                 [[r["lambda"], r["retained"], r["revoked"], r["infer_acc"]] for r in rows])


def plot_roc(curves, path, csv_path=None):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    rows = []
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        rows += [[name, f, t] for f, t in zip(fpr, tpr)]
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Unlearning-detection ROC (lower = better)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    if csv_path:
        save_csv(csv_path, ["method", "fpr", "tpr"], rows)


def plot_ablations(rows, path, csv_path=None):
    names = [r["ablation"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, ttl in [(axes[0], "retained", "retained utility (higher=better)"),
                         (axes[1], "revoked", "revoked return (lower=forget)"),
                         (axes[2], "infer_acc", "infer acc (<=0.5 private)")]:
        vals = [r[key] for r in rows]
        ax.bar(names, vals, color="#4c72b0")
        if key == "infer_acc":
            ax.axhline(0.5, color="gray", ls=":")
        ax.set_title(ttl, fontsize=9)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    if csv_path:
        save_csv(csv_path, ["ablation", "retained", "revoked", "infer_acc", "seconds"],
                 [[r["ablation"], r["retained"], r["revoked"], r["infer_acc"], r["seconds"]]
                  for r in rows])


def plot_algos(rows, path, csv_path=None):
    names = [r["algo"] for r in rows]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, key, ttl in [(axes[0], "retained", "retained utility"),
                         (axes[1], "revoked", "revoked return"),
                         (axes[2], "infer_acc", "infer acc"),
                         (axes[3], "speedup", "retrain/unlearn speedup")]:
        ax.bar(names, [r[key] for r in rows], color="#55a868")
        if key == "infer_acc":
            ax.axhline(0.5, color="gray", ls=":")
        ax.set_title(ttl, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    if csv_path:
        save_csv(csv_path, ["algo", "retained", "revoked", "infer_acc", "speedup"],
                 [[r["algo"], r["retained"], r["revoked"], r["infer_acc"], r["speedup"]]
                  for r in rows])
