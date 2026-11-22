"""GRPO для ответа из одного токена и SFT-контроль.

Группа из G ответов сэмплируется из одного распределения за один прямой проход.
Обновление одно на порцию сэмплов, поэтому PPO-клиппинг ничего бы не делал, его нет.
KL считается точно, суммой по k+1 вариантам.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray


def sample_actions(
    logp: NDArray[np.float32], group_size: int, temperature: float, rng: np.random.Generator
) -> NDArray[np.int64]:
    """[B, k+1] в [B, G]: G ответов на промпт из policy^(1/T)."""
    scaled = logp / temperature
    probs = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    return np.stack([rng.choice(len(p), size=group_size, p=p) for p in probs])


def group_advantages(rewards: NDArray[np.float32]) -> NDArray[np.float32]:
    """(r - mean) / std по группе. Все награды одинаковые - advantage 0."""
    mean = rewards.mean(axis=1, keepdims=True)
    std = rewards.std(axis=1, keepdims=True)
    adv = np.where(std > 1e-6, (rewards - mean) / np.maximum(std, 1e-6), 0.0)
    return adv.astype(np.float32)


def kl_to_reference(logp: mx.array, ref_logp: mx.array) -> mx.array:
    """KL(π || π_ref) по каждому промпту, [B, k+1] в [B]."""
    return (mx.exp(logp) * (logp - ref_logp)).sum(axis=-1)


def grpo_loss(
    logp: mx.array,
    ref_logp: mx.array,
    actions: mx.array,
    advantages: mx.array,
    kl_coef: float,
) -> tuple[mx.array, mx.array]:
    """(loss, средний KL). actions и advantages формы [B, G]."""
    chosen = mx.take_along_axis(logp, actions, axis=1)
    policy_term = -(advantages * chosen).mean()
    kl = kl_to_reference(logp, ref_logp).mean()
    return policy_term + kl_coef * kl, kl


def sft_loss(logp: mx.array, ref_logp: mx.array, targets: mx.array) -> tuple[mx.array, mx.array]:
    """Кросс-энтропия на правильную цифру. KL в loss не входит, чисто для лога."""
    nll = -mx.take_along_axis(logp, targets[:, None], axis=1).mean()
    return nll, kl_to_reference(logp, ref_logp).mean()


def mean_kl(logp: NDArray[np.float32], ref_logp: NDArray[np.float32]) -> float:
    """``kl_to_reference`` в numpy, среднее по промптам."""
    return float((np.exp(logp) * (logp - ref_logp)).sum(axis=1).mean())


def entropy(logp: NDArray[np.float32]) -> float:
    """Средняя энтропия, в натах."""
    return float(-(np.exp(logp) * logp).sum(axis=1).mean())
