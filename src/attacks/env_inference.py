"""Environment Inference Attack (Ye et al. NDSS'25, object-level).

Adversary observes rollouts inside the REVOKED sector and must decide whether
that environment shaped the policy. Protocol:
  train on (seen=original, unseen=oracle-never-trained),
  then test on (unlearned, unseen):  ~50%  => certified forgetting.
"""
import numpy as np
import torch
import torch.nn as nn


def rollout_features(agent, env, sector, n, seed, cfg):
    rng = np.random.default_rng(seed + 777)
    feats = []
    for _ in range(n):
        s = env.reset(rng, sector=sector)
        ret, reach, speeds, anorms, turns = 0.0, False, [], [], []
        prev_a = np.zeros(cfg.action_dim)
        for _ in range(cfg.episode_len):
            a = agent.act(s) if agent is not None else rng.uniform(-1, 1, cfg.action_dim).astype(np.float32)
            s2, r, done, info = env.step(a)
            ret += r
            reach = reach or info["reach"]
            speeds.append(np.linalg.norm(s[2:4]))
            anorms.append(np.linalg.norm(a))
            turns.append(np.linalg.norm(a - prev_a))
            prev_a = a
            s = s2
            if done:
                break
        feats.append([ret, float(reach), np.mean(speeds), np.mean(anorms),
                      np.std(anorms), np.mean(turns)])
    return np.array(feats, np.float32)


class Attacker(nn.Module):
    def __init__(self, dim=6, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_attacker(fA, fB, device, seed=0, epochs=300, lr=1e-3):
    torch.manual_seed(seed)
    X = np.concatenate([fA, fB], 0)
    y = np.concatenate([np.ones(len(fA)), np.zeros(len(fB))]).astype(np.float32)
    perm = np.random.default_rng(seed).permutation(len(X))
    X, y = X[perm], y[perm]
    Xt = torch.as_tensor(X, device=device)
    yt = torch.as_tensor(y, device=device).unsqueeze(1)
    model = Attacker(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        idx = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), 64):
            b = idx[i:i + 64]
            loss = nn.functional.binary_cross_entropy_with_logits(model(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def pair_accuracy(model, fA, fB, device):
    X = np.concatenate([fA, fB], 0)
    y = np.concatenate([np.ones(len(fA)), np.zeros(len(fB))])
    p = torch.sigmoid(model(torch.as_tensor(X, device=device))).cpu().numpy().ravel()
    pred = (p > 0.5).astype(float)
    return float((pred == y).mean())


@torch.no_grad()
def pair_auc(model, fA, fB, device):
    pA = torch.sigmoid(model(torch.as_tensor(fA, device=device))).cpu().numpy().ravel()
    pB = torch.sigmoid(model(torch.as_tensor(fB, device=device))).cpu().numpy().ravel()
    return float((pA[:, None] > pB[None, :]).mean() + 0.5 * (pA[:, None] == pB[None, :]).mean())

@torch.no_grad()
def pair_roc(model, fA, fB, device):
    """ROC curve treating fA as positives. Returns (fpr, tpr)."""
    import numpy as np
    pA = torch.sigmoid(model(torch.as_tensor(fA, device=device))).cpu().numpy().ravel()
    pB = torch.sigmoid(model(torch.as_tensor(fB, device=device))).cpu().numpy().ravel()
    thr = np.sort(np.concatenate([pA, pB]))[::-1]
    fpr = np.array([(pB >= t).mean() for t in thr])
    tpr = np.array([(pA >= t).mean() for t in thr])
    return fpr, tpr
