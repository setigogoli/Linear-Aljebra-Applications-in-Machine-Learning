"""Small NumPy models with explicit gradients and deterministic initialization."""

import numpy as np
from scipy.special import expit, logsumexp


def softmax(z):
    """Normalize each row, including logits with large magnitudes."""
    z = np.asarray(z)
    return np.exp(z - logsumexp(z, axis=-1, keepdims=True))


def binary_loss(logits, y):
    return float(np.mean(np.logaddexp(0, logits) - np.asarray(y) * logits))


def least_squares(x, y):
    """Solve a zero-intercept least-squares problem without forming an inverse."""
    return float(np.linalg.lstsq(np.asarray(x)[:, None], y, rcond=None)[0][0])


class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=3000):
        self.learning_rate, self.epochs = learning_rate, epochs

    def fit(self, x, y):
        y = np.asarray(y)
        self.w, self.b, self.losses = np.zeros(x.shape[1]), 0.0, []
        for _ in range(self.epochs):
            z = x @ self.w + self.b
            error = expit(z) - y
            self.losses.append(binary_loss(z, y))
            self.w -= self.learning_rate * (x.T @ error / len(x))
            self.b -= self.learning_rate * error.mean()
        return self

    def predict(self, x):
        return (expit(x @ self.w + self.b) >= 0.5).astype(int)


class SoftmaxRegression:
    def __init__(self, learning_rate=0.1, epochs=1500, seed=42):
        self.learning_rate, self.epochs, self.seed = learning_rate, epochs, seed

    def fit(self, x, y):
        self.classes = np.unique(y)
        target = np.eye(len(self.classes))[np.searchsorted(self.classes, y)]
        self.w = np.random.default_rng(self.seed).normal(0, 0.01, (x.shape[1], len(self.classes)))
        self.b = np.zeros(len(self.classes))
        self.losses = []
        for _ in range(self.epochs):
            z = x @ self.w + self.b
            self.losses.append(float(np.mean(logsumexp(z, axis=1) - np.sum(target * z, axis=1))))
            dz = (softmax(z) - target) / len(x)
            self.w -= self.learning_rate * x.T @ dz
            self.b -= self.learning_rate * dz.sum(axis=0)
        return self

    def predict(self, x):
        return self.classes[np.argmax(x @ self.w + self.b, axis=1)]


class MLP:
    """One ReLU hidden layer and a binary output."""
    def __init__(self, input_dim=2, hidden_dim=16, seed=42):
        rng = np.random.default_rng(seed)
        self.params = {
            'w1': rng.normal(size=(input_dim, hidden_dim)) * np.sqrt(2 / input_dim),
            'b1': np.zeros(hidden_dim),
            'w2': rng.normal(size=(hidden_dim, 1)) / np.sqrt(hidden_dim),
            'b2': np.zeros(1),
        }

    def loss_grad(self, x, y):
        p = self.params
        z1 = x @ p['w1'] + p['b1']
        a1 = np.maximum(z1, 0)
        z2 = a1 @ p['w2'] + p['b2']
        y = np.asarray(y).reshape(-1, 1)
        dz2 = (expit(z2) - y) / len(x)
        dz1 = (dz2 @ p['w2'].T) * (z1 > 0)
        return binary_loss(z2, y), {
            'w1': x.T @ dz1, 'b1': dz1.sum(axis=0),
            'w2': a1.T @ dz2, 'b2': dz2.sum(axis=0),
        }

    def fit(self, x, y, epochs=5000, learning_rate=0.1):
        self.losses = []
        for _ in range(epochs):
            loss, gradients = self.loss_grad(x, y)
            self.losses.append(loss)
            for k in self.params:
                self.params[k] -= learning_rate * gradients[k]
        return self

    def predict(self, x):
        p = self.params
        z = np.maximum(x @ p['w1'] + p['b1'], 0) @ p['w2'] + p['b2']
        return (z.ravel() >= 0).astype(int)


class CharacterRNN:
    """A tanh recurrent network trained by truncated BPTT and Adagrad."""
    def __init__(self, vocab_size, hidden_size=48, seed=42):
        self.hidden_size, self.vocab_size = hidden_size, vocab_size
        rng = np.random.default_rng(seed)
        self.params = {
            'wx': rng.normal(0, .05, (hidden_size, vocab_size)),
            'wh': rng.normal(0, .05, (hidden_size, hidden_size)),
            'wy': rng.normal(0, .05, (vocab_size, hidden_size)),
            'bh': np.zeros(hidden_size), 'by': np.zeros(vocab_size),
        }
        self.memory = {k: np.zeros_like(v) for k, v in self.params.items()}

    def loss_grad(self, inputs, targets, h0=None):
        p = self.params
        h = [np.zeros(self.hidden_size) if h0 is None else h0.copy()]
        probabilities, loss = [], 0.0
        for x, target in zip(inputs, targets):
            h.append(np.tanh(p['wx'][:, x] + p['wh'] @ h[-1] + p['bh']))
            z = p['wy'] @ h[-1] + p['by']
            probabilities.append(softmax(z))
            loss += logsumexp(z) - z[target]
        grad = {k: np.zeros_like(v) for k, v in p.items()}
        dh_next = np.zeros(self.hidden_size)
        for t in reversed(range(len(inputs))):
            dy = probabilities[t].copy()
            dy[targets[t]] -= 1
            dy /= len(inputs)
            grad['wy'] += np.outer(dy, h[t + 1])
            grad['by'] += dy
            dh = p['wy'].T @ dy + dh_next
            dz = dh * (1 - h[t + 1] ** 2)
            grad['wx'][:, inputs[t]] += dz
            grad['wh'] += np.outer(dz, h[t])
            grad['bh'] += dz
            dh_next = p['wh'].T @ dz
        return float(loss / len(inputs)), grad, h[-1]

    def fit(self, ids, steps=2000, sequence_length=32, learning_rate=0.08, seed=42):
        rng = np.random.default_rng(seed)
        self.losses = []
        for _ in range(steps):
            start = rng.integers(0, len(ids) - sequence_length)
            # Independent random windows must start with independent zero states.
            loss, gradients, _ = self.loss_grad(ids[start:start+sequence_length], ids[start+1:start+sequence_length+1])
            self.losses.append(loss)
            for k, g in gradients.items():
                g = np.clip(g, -5, 5)
                self.memory[k] += g * g
                self.params[k] -= learning_rate * g / np.sqrt(self.memory[k] + 1e-8)
        return self

    def sample(self, seed_index, length=400, temperature=0.8, seed=123):
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        rng = np.random.default_rng(seed)
        h, x, out = np.zeros(self.hidden_size), seed_index, []
        p = self.params
        for _ in range(length):
            h = np.tanh(p['wx'][:, x] + p['wh'] @ h + p['bh'])
            probabilities = softmax((p['wy'] @ h + p['by']) / temperature)
            x = int(rng.choice(self.vocab_size, p=probabilities))
            out.append(x)
        return out


class LSTMClassifier:
    """Frozen word embeddings, a trainable LSTM, and a binary output in NumPy.

    Inputs have shape (batch, time, embedding_dim); lengths exclude right padding.
    The padding mask preserves both recurrent states and their backward gradients.
    """
    def __init__(self, input_dim, hidden_dim=24, seed=42):
        rng = np.random.default_rng(seed)
        self.hidden_dim = hidden_dim
        self.params = {
            'w': rng.normal(0, 1 / np.sqrt(input_dim + hidden_dim), (input_dim + hidden_dim, 4 * hidden_dim)),
            'b': np.zeros(4 * hidden_dim),
            'v': rng.normal(0, 1 / np.sqrt(hidden_dim), hidden_dim),
            'bias': np.zeros(1),
        }
        self.params['b'][:hidden_dim] = 1.0  # Forget-gate bias.
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.step = 0

    def forward(self, x, lengths):
        p, d = self.params, self.hidden_dim
        h, c = np.zeros((len(x), d)), np.zeros((len(x), d))
        cache = []
        for t in range(x.shape[1]):
            joint = np.concatenate([x[:, t], h], axis=1)
            gates = joint @ p['w'] + p['b']
            f, i, o = np.split(expit(gates[:, :3*d]), 3, axis=1)
            g = np.tanh(gates[:, 3*d:])
            c_new = f * c + i * g
            h_new = o * np.tanh(c_new)
            mask = (t < lengths)[:, None]
            cache.append((joint, c, f, i, o, g, c_new, mask))
            c = np.where(mask, c_new, c)
            h = np.where(mask, h_new, h)
        return h @ p['v'] + p['bias'][0], h, cache

    def loss_grad(self, x, lengths, y):
        p = self.params
        logits, h, cache = self.forward(x, lengths)
        dz = (expit(logits) - y) / len(y)
        gradients = {k: np.zeros_like(v) for k, v in p.items()}
        gradients['v'] = h.T @ dz
        gradients['bias'][0] = dz.sum()
        dh = dz[:, None] * p['v']
        dc = np.zeros_like(dh)
        for joint, c_old, f, i, o, g, c_new, mask in reversed(cache):
            dh_active, dc_active = dh * mask, dc * mask
            dc_total = dc_active + dh_active * o * (1 - np.tanh(c_new) ** 2)
            df = dc_total * c_old * f * (1 - f)
            di = dc_total * g * i * (1 - i)
            do = dh_active * np.tanh(c_new) * o * (1 - o)
            dg = dc_total * i * (1 - g ** 2)
            dgates = np.concatenate([df, di, do, dg], axis=1)
            gradients['w'] += joint.T @ dgates
            gradients['b'] += dgates.sum(axis=0)
            dh = dh * (~mask) + (dgates @ p['w'].T)[:, x.shape[2]:]
            dc = dc * (~mask) + dc_total * f
        return binary_loss(logits, y), gradients

    def update(self, gradients, learning_rate=0.003):
        self.step += 1
        norm = np.sqrt(sum(np.sum(g*g) for g in gradients.values()))
        scale = min(1.0, 5 / (norm + 1e-12))
        for k, gradient in gradients.items():
            gradient = gradient * scale
            self.m[k] = .9 * self.m[k] + .1 * gradient
            self.v[k] = .999 * self.v[k] + .001 * gradient * gradient
            m = self.m[k] / (1 - .9 ** self.step)
            v = self.v[k] / (1 - .999 ** self.step)
            self.params[k] -= learning_rate * m / (np.sqrt(v) + 1e-8)


class MatrixFactorization:
    """Observed-entry factorization with global, user, and item biases.

    SGD minimizes one half of squared error plus per-observation L2 penalty.
    The global mean is fixed from training data. Unseen factors start at zero.
    """
    def __init__(self, n_users, n_items, rank=20, seed=42):
        rng = np.random.default_rng(seed)
        self.p = rng.normal(0, .1, (n_users, rank))
        self.q = rng.normal(0, .1, (n_items, rank))
        self.bu, self.bi = np.zeros(n_users), np.zeros(n_items)
        self.mean = 0.0

    def initialize(self, users, items, ratings):
        self.mean = float(np.mean(ratings))
        self.p[~np.isin(np.arange(len(self.p)), np.unique(users))] = 0
        self.q[~np.isin(np.arange(len(self.q)), np.unique(items))] = 0

    def predict(self, users, items):
        return self.mean + self.bu[users] + self.bi[items] + np.sum(self.p[users] * self.q[items], axis=-1)

    def epoch(self, users, items, ratings, rng, learning_rate=0.005, regularization=0.02):
        for j in rng.permutation(len(ratings)):
            u, i, r = users[j], items[j], ratings[j]
            pu, qi = self.p[u].copy(), self.q[i].copy()
            error = r - (self.mean + self.bu[u] + self.bi[i] + pu @ qi)
            self.bu[u] += learning_rate * (error - regularization * self.bu[u])
            self.bi[i] += learning_rate * (error - regularization * self.bi[i])
            self.p[u] += learning_rate * (error * qi - regularization * pu)
            self.q[i] += learning_rate * (error * pu - regularization * qi)
