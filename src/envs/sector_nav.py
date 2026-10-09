"""Continuous navigation MDP partitioned into spatial sub-environments (sectors).

Each sector has a DISTINCT drift (wind) field => different transition dynamics,
which is exactly what an environment-inference adversary tries to recover.
State  : [x, y, vx, vy, gx, gy]  in [0,1]^2 x R^2 x [0,1]^2
Action : continuous force in [-1, 1]^2
Sector : quadrant of the current position (0..3). Sector 0 = revoked.
"""
import numpy as np

DRIFTS = np.array([
    [0.15, 0.05],
    [-0.10, 0.12],
    [0.08, -0.14],
    [-0.12, -0.08],
], dtype=np.float32)


class SectorNavEnv:
    def __init__(self, n_sectors=4, episode_len=100, dt=0.05, friction=0.90):
        self.n_sectors = n_sectors
        self.episode_len = episode_len
        self.dt = dt
        self.friction = friction
        self.drifts = DRIFTS[:n_sectors]

    # ---- helpers ----
    def sector_of(self, xy):
        x, y = xy
        return int(x >= 0.5) + 2 * int(y >= 0.5)

    def _quadrant_bounds(self, k):
        x0 = 0.5 if k % 2 == 1 else 0.0
        y0 = 0.5 if k >= 2 else 0.0
        return x0, y0

    # ---- API ----
    def reset(self, rng, sector=None):
        sector = int(rng.integers(self.n_sectors)) if sector is None else int(sector)
        x0, y0 = self._quadrant_bounds(sector)
        self.xy = rng.uniform([x0 + 0.05, y0 + 0.05], [x0 + 0.45, y0 + 0.45]).astype(np.float32)
        self.goal = rng.uniform([x0 + 0.05, y0 + 0.05], [x0 + 0.45, y0 + 0.45]).astype(np.float32)
        while np.linalg.norm(self.goal - self.xy) < 0.20:
            self.goal = rng.uniform([x0 + 0.05, y0 + 0.05], [x0 + 0.45, y0 + 0.45]).astype(np.float32)
        self.v = np.zeros(2, dtype=np.float32)
        self.t = 0
        self.episode_sector = sector
        return self._obs()

    def _obs(self):
        return np.concatenate([self.xy, self.v, self.goal]).astype(np.float32)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        k = self.sector_of(self.xy)
        self.v = self.friction * self.v + (a + self.drifts[k]) * self.dt
        self.xy = np.clip(self.xy + self.v * self.dt, 0.0, 1.0)
        self.t += 1

        dist = float(np.linalg.norm(self.goal - self.xy))
        reach = dist < 0.05
        reward = -0.01 - 0.5 * dist
        if reach:
            reward += 1.0
        done = reach or (self.t >= self.episode_len)
        return self._obs(), reward, done, {"sector": k, "reach": reach}
