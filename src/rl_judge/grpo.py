"""GRPO для ответа из одного токена (и SFT-контроль той же формы).

Обычный GRPO на каждый промпт генерирует G ответов, раздаёт им награды и
сравнивает каждый ответ со средним по своей группе: advantage = (r − mean) / std.
Потом повышает вероятность ответов лучше среднего, понижает — хуже, а KL к
исходной модели не даёт уйти от неё слишком далеко.

У нас ответ — одна цифра, поэтому G ответов не нужно генерировать заново: они
сэмплируются из одного распределения, полученного за один прямой проход. Всё
остальное — как в GRPO. Обновление одно на каждую порцию сэмплов, так что
отношение вероятностей новой и старой политики равно 1 и PPO-клиппинг ничего не
делает — его и нет. KL считаем точно, суммой по k+1 вариантам, а не оценкой по
сэмплам, как в исходной статье: вариантов мало, и точное значение дешевле.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray


def sample_actions(
    logp: NDArray[np.float32], group_size: int, temperature: float, rng: np.random.Generator
) -> NDArray[np.int64]:
    """G ответов на каждый промпт из распределения policy^(1/T): [B, k+1] → [B, G]."""
    scaled = logp / temperature
    probs = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    return np.stack([rng.choice(len(p), size=group_size, p=p) for p in probs])


def group_advantages(rewards: NDArray[np.float32]) -> NDArray[np.float32]:
    """(r − среднее по группе) / std группы. Если все ответы группы получили одно
    и то же, сигнала нет — advantage нулевой (а не деление на ноль)."""
    mean = rewards.mean(axis=1, keepdims=True)
    std = rewards.std(axis=1, keepdims=True)
    adv = np.where(std > 1e-6, (rewards - mean) / np.maximum(std, 1e-6), 0.0)
    return adv.astype(np.float32)


def kl_to_reference(logp: mx.array, ref_logp: mx.array) -> mx.array:
    """KL(π ‖ π_ref) по вариантам ответа, по каждому промпту: [B, k+1] → [B]."""
    return (mx.exp(logp) * (logp - ref_logp)).sum(axis=-1)


def grpo_loss(
    logp: mx.array,
    ref_logp: mx.array,
    actions: mx.array,
    advantages: mx.array,
    kl_coef: float,
) -> tuple[mx.array, mx.array]:
    """Возвращает (loss, средний KL). actions и advantages — [B, G]."""
    chosen = mx.take_along_axis(logp, actions, axis=1)
    policy_term = -(advantages * chosen).mean()
    kl = kl_to_reference(logp, ref_logp).mean()
    return policy_term + kl_coef * kl, kl


def sft_loss(logp: mx.array, ref_logp: mx.array, targets: mx.array) -> tuple[mx.array, mx.array]:
    """Кросс-энтропия на правильную цифру (0, если верного кода в списке нет).
    KL в loss не входит, но возвращается для лога — чтобы сравнивать с GRPO."""
    nll = -mx.take_along_axis(logp, targets[:, None], axis=1).mean()
    return nll, kl_to_reference(logp, ref_logp).mean()


def mean_kl(logp: NDArray[np.float32], ref_logp: NDArray[np.float32]) -> float:
    """То же, что ``kl_to_reference``, но в numpy и усреднённое: для оценок без градиента."""
    return float((np.exp(logp) * (logp - ref_logp)).sum(axis=1).mean())


def entropy(logp: NDArray[np.float32]) -> float:
    """Средняя энтропия политики, в натах. Падает до нуля — модель «застыла»."""
    return float(-(np.exp(logp) * logp).sum(axis=1).mean())
