"""Binary-tree prefix-sum mechanism + maximal-coupling unlearning (Theorem 3).

Each node stores (clean v, noisy v~ = v + xi), xi ~ N(0, sigma^2 I).
Unlearning leaf t with replacement statistic x' traverses leaf->root and
maximally couples N(v, s2 I) -> N(v + x' - x, s2 I) via rejection sampling
with the Gaussian reflection map on rejection.
"""
import numpy as np


class BinaryTree:
    def __init__(self, cap, dim, sigma, rng):
        self.cap = int(cap)
        self.dim = dim
        self.sigma = float(sigma)
        self.rng = rng
        self.clean = np.zeros((2 * self.cap, dim), np.float64)
        self.noisy = np.zeros((2 * self.cap, dim), np.float64)
        self.leaf = np.zeros((self.cap, dim), np.float64)
        self.used = np.zeros(self.cap, bool)
        xi = rng.normal(0.0, sigma, (2 * self.cap, dim))
        self._noise = xi

    def tree_update(self, x, t):
        x = np.asarray(x, np.float64)
        delta = x - (self.leaf[t] if self.used[t] else 0.0)
        self.leaf[t] = x
        self.used[t] = True
        node = self.cap + t
        while node >= 1:
            self.clean[node] += delta
            self.noisy[node] = self.clean[node] + self._noise[node]
            node //= 2

    def unlearn(self, t, x_new):
        """Maximal coupling along path. Returns (retrain, accepts, path_len)."""
        x_new = np.asarray(x_new, np.float64)
        delta = x_new - self.leaf[t]
        node = self.cap + t
        retrain, accepts, path = False, 0, 0
        s2 = 2.0 * self.sigma ** 2
        while node >= 1:
            v = self.clean[node].copy()
            v_new = v + delta
            vt = self.noisy[node]
            log_ratio = (-(np.sum((vt - v_new) ** 2)) + np.sum((vt - v) ** 2)) / s2
            if self.rng.uniform() <= min(1.0, np.exp(log_ratio)):
                self.clean[node] = v_new          # accept: reuse old noise
                accepts += 1
            else:                                  # reject: reflection map
                self.noisy[node] = v_new + v - vt
                self.clean[node] = v_new
                retrain = True
                path += 1
                break
            path += 1
            node //= 2
        self.leaf[t] = x_new
        return retrain, accepts, path
