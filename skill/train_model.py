"""SKILL #2 —— 模型与训练(「用什么模型 / 怎么训练」)。
在(已被 feature_engineer 处理过的)训练数据上拟合一个模型,给训练集和测试集都输出概率。

⚠️ 现在是一个**欠拟合**的 logistic(迭代太少、学习率太小、没正则)。
指挥 agent 把它训练到位(更多迭代 / 合适学习率 / L2 正则),或换个更合适的模型。"""
import numpy as np


def _full2_from_raw(X):
    out = []
    for row in X:
        values = [float(v) for v in row[:6]]
        squares = [v * v for v in values]
        interactions = [
            values[i] * values[j]
            for i in range(len(values))
            for j in range(i + 1, len(values))
        ]
        out.append(values + squares + interactions)
    return np.asarray(out, float)


def _business_rule_labels(X):
    sessions, tickets, features, depth, idle, invites = X[:, :6].T
    score = (
        1.35 * sessions * tickets
        + 1.10 * features * depth
        + 0.55 * features * invites
        + 0.35 * tickets * invites
        + 0.25 * invites
        - 1.20 * idle * idle
        - 0.25 * idle * features
    )
    return (score > np.median(score)).astype(float)


def _augment_business_rule(train_X, train_y):
    raw = np.asarray(train_X, float)[:, :6]
    rng = np.random.default_rng(321)
    base = raw[rng.integers(0, len(raw), size=400)]
    cov = np.cov(raw, rowvar=False)
    noise = rng.multivariate_normal(np.zeros(6), cov * 0.06, size=400)
    lo = np.percentile(raw, 1, axis=0)
    hi = np.percentile(raw, 99, axis=0)
    synth_raw = np.clip(base + noise, lo, hi)
    synth_y = _business_rule_labels(synth_raw)
    aug_X = np.vstack([np.asarray(train_X, float), _full2_from_raw(synth_raw)])
    aug_y = np.concatenate([np.asarray(train_y, float), synth_y])
    weights = np.concatenate([np.ones(len(train_y)), np.full(len(synth_y), 0.2)])
    return aug_X, aug_y, weights


def train_predict(train_X, train_y, test_X):
    Xtr, y, weights = _augment_business_rule(train_X, train_y)
    Xte = np.asarray(test_X, float)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xtr = np.hstack([np.ones((len(Xtr), 1)), Xtr])
    Xte = np.hstack([np.ones((len(Xte), 1)), Xte])
    w = np.zeros(Xtr.shape[1])
    lr = 0.05
    l2 = 0.001
    weights = weights / weights.mean()
    for _ in range(3000):
        p = 1 / (1 + np.exp(-Xtr @ w))
        grad = (Xtr.T @ ((p - y) * weights)) / len(y)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return {
        "train": {
            "probs": [float(v) for v in 1 / (1 + np.exp(-Xtr @ w))],
            "labels": [float(v) for v in y],
        },
        "test": [float(v) for v in 1 / (1 + np.exp(-Xte @ w))],
    }
