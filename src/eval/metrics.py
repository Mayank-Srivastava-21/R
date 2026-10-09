import numpy as np


def evaluate(agent, env, sector, n, seed, cfg):
    rng = np.random.default_rng(seed + 31337)
    rets, reaches = [], []
    for _ in range(n):
        s = env.reset(rng, sector=sector)
        ret, reach = 0.0, False
        for _ in range(cfg.episode_len):
            a = agent.act(s) if agent is not None else rng.uniform(-1, 1, cfg.action_dim).astype(np.float32)
            s, r, done, info = env.step(a)
            ret += r
            reach = reach or info["reach"]
            if done:
                break
        rets.append(ret); reaches.append(reach)
    return float(np.mean(rets)), float(np.mean(reaches))
