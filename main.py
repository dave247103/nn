import numpy as np


def generate_dataset(modes0, modes1, spm, mean_range=(-1, 1), var_range=(0.5, 2.0), seed=None):
    rng = np.random.default_rng(seed)

    def gen_class(modes, label):
        if modes <= 0 or spm <= 0:
            return np.empty((0, 2)), np.empty((0,), dtype=int)
        xs = []
        for _ in range(modes):
            mean = rng.uniform(mean_range[0], mean_range[1], size=2)
            var = rng.uniform(var_range[0], var_range[1], size=2)
            std = np.sqrt(var)
            xs.append(rng.normal(loc=mean, scale=std, size=(spm, 2)))
        Xc = np.vstack(xs) if xs else np.empty((0, 2))
        yc = np.full((Xc.shape[0],), label, dtype=int)
        return Xc, yc

    X0, y0 = gen_class(modes0, 0)
    X1, y1 = gen_class(modes1, 1)
    X = np.vstack([X0, X1]) if X0.size or X1.size else np.empty((0, 2))
    y = np.concatenate([y0, y1]) if y0.size or y1.size else np.empty((0,), dtype=int)
    if len(y) == 0:
        return X, y
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def train_test_split(X, y, test_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    if n == 0:
        return X, X, y, y
    idx = rng.permutation(n)
    n_test = int(round(n * test_ratio))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class NeuralNet:
    def __init__(self, layer_sizes, seed=0):
        self.layer_sizes = list(layer_sizes)
        rng = np.random.default_rng(seed)
        self.W = []
        self.b = []
        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            w = rng.normal(0.0, 1.0 / np.sqrt(max(1, n_in)), size=(n_in, n_out))
            b = np.zeros((n_out,), dtype=float)
            self.W.append(w)
            self.b.append(b)

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _sigmoid_prime(z):
        s = 1.0 / (1.0 + np.exp(-z))
        return s * (1.0 - s)

    def forward(self, X):
        A = X
        As = [A]
        Zs = []
        for W, b in zip(self.W, self.b):
            Z = A @ W + b
            A = self._sigmoid(Z)
            Zs.append(Z)
            As.append(A)
        return As, Zs

    def predict_proba(self, X):
        As, _ = self.forward(X)
        return As[-1]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def train(self, X, y, eta, epochs, batch_size=None):
        n = len(y)
        if n == 0:
            return
        y = y.astype(int)
        D = np.zeros((n, 2), dtype=float)
        D[np.arange(n), y] = 1.0
        for _ in range(int(epochs)):
            idx = np.random.permutation(n)
            Xs = X[idx]
            Ds = D[idx]
            if not batch_size or batch_size <= 0 or batch_size >= n:
                batches = [(Xs, Ds)]
            else:
                bs = int(batch_size)
                batches = [(Xs[i:i + bs], Ds[i:i + bs]) for i in range(0, n, bs)]
            for Xb, Db in batches:
                As, Zs = self.forward(Xb)
                deltas = [None] * len(self.W)
                deltas[-1] = (Db - As[-1]) * self._sigmoid_prime(Zs[-1])
                for k in range(len(self.W) - 2, -1, -1):
                    deltas[k] = (deltas[k + 1] @ self.W[k + 1].T) * self._sigmoid_prime(Zs[k])
                m = len(Xb)
                for k in range(len(self.W)):
                    self.W[k] += eta * (As[k].T @ deltas[k]) / m
                    self.b[k] += eta * deltas[k].mean(axis=0)
