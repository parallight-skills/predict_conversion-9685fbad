"""SKILL #3 —— 校准 / 预测(「如何 predict」)。
把模型输出的概率变成最终的 0/1 标签。

⚠️ 现在是固定 0.5 阈值。指挥 agent 改成**在训练集上挑一个让训练 accuracy 最高的阈值**
(threshold tuning)——类别不均衡时,这一步能再抠几分。这是价值最小、但也最便宜的一件 skill。"""


def to_labels(train_probs, train_y, test_probs):
    if isinstance(train_probs, dict):
        train_y = train_probs["labels"]
        train_probs = train_probs["probs"]
    candidates = sorted(set(float(p) for p in train_probs))
    if not candidates:
        return []
    thr = max(
        candidates,
        key=lambda t: sum((p >= t) == bool(y) for p, y in zip(train_probs, train_y)),
    )
    return [1 if p >= thr else 0 for p in test_probs]
