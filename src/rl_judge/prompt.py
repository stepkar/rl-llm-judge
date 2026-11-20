"""Текст промпта и связь «цифра ответа ↔ код кандидата».

Кандидаты показываются в перемешанном порядке. Если оставить порядок CatBoost,
модель быстро выучит «всегда отвечай 1» (это уже 68% верных) и мы будем мерить
не понимание текста, а умение копировать ранжировщик. Перестановка у каждой
задачи своя, но фиксированная (зависит от seed и id), поэтому базовая и все
обученные модели видят один и тот же промпт.

Ответ — одна цифра: 1..k — номер варианта, 0 — «подходящего нет».
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from rl_judge.data import Example

ABSTAIN = 0


@dataclass(frozen=True, slots=True)
class Prompt:
    example: Example
    order: tuple[int, ...]  # order[j] — индекс кандидата (в порядке CatBoost) на месте j+1
    text: str  # сообщение пользователя; chat-шаблон модели накладывается в model.py

    @property
    def n_options(self) -> int:
        return len(self.order) + 1  # плюс «0 — нет подходящего»

    def code_of(self, action: int) -> str | None:
        """Код, который выбрала модель; None — воздержалась."""
        if action == ABSTAIN:
            return None
        return self.example.candidates[self.order[action - 1]].code

    def gold_action(self) -> int:
        """Правильная цифра: место верного кода в списке или 0, если его там нет."""
        for j, idx in enumerate(self.order):
            if self.example.candidates[idx].code == self.example.gold:
                return j + 1
        return ABSTAIN


def shuffled_order(example: Example, k: int, seed: int) -> tuple[int, ...]:
    # Пул каскада — 60–80 кодов, так что короче k он не бывает; если всё же да,
    # лучше упасть сразу, чем тихо показывать модели разное число вариантов.
    if len(example.candidates) < k:
        raise ValueError(
            f"у задачи {example.id} всего {len(example.candidates)} кандидатов, нужно {k}"
        )
    digest = hashlib.sha1(f"{seed}:{example.id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return tuple(int(i) for i in rng.permutation(k))


def _unit(unit: str | None) -> str:
    return f" [{unit}]" if unit else ""


def render(example: Example, order: tuple[int, ...]) -> str:
    lines = [
        "Нужно подобрать код из строительного справочника для позиции из заявки организации.",
        "Тексты уже нормализованы: нижний регистр, без ГОСТов, артикулов и упаковки.",
        "",
        f"Позиция: {example.query}{_unit(example.unit)}",
        "",
        "Варианты из справочника:",
    ]
    for j, idx in enumerate(order, start=1):
        cand = example.candidates[idx]
        lines.append(f"{j}. {cand.name}{_unit(cand.unit)}")
    lines.append(f"{ABSTAIN}. Ни один вариант не подходит")
    lines += ["", "Ответь одной цифрой — номером подходящего варианта."]
    return "\n".join(lines)


def build_prompts(examples: list[Example], k: int, seed: int) -> list[Prompt]:
    prompts = []
    for ex in examples:
        order = shuffled_order(ex, k, seed)
        prompts.append(Prompt(example=ex, order=order, text=render(ex, order)))
    return prompts
