"""Промпт и перевод цифры ответа в код кандидата.

Кандидаты перемешаны: в порядке CatBoost модель выучила бы "всегда отвечай 1",
это уже 68% верных. Перестановка зависит от seed и id, так что все модели видят
один и тот же промпт. Ответ: 1..k номер варианта, 0 - подходящего нет.
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
    order: tuple[int, ...]  # order[j] - индекс кандидата у CatBoost на месте j+1
    text: str  # без chat-шаблона, он добавляется в model.py

    @property
    def n_options(self) -> int:
        return len(self.order) + 1

    def code_of(self, action: int) -> str | None:
        if action == ABSTAIN:
            return None
        return self.example.candidates[self.order[action - 1]].code

    def gold_action(self) -> int:
        """Правильная цифра, 0 если верного кода в списке нет."""
        for j, idx in enumerate(self.order):
            if self.example.candidates[idx].code == self.example.gold:
                return j + 1
        return ABSTAIN


def shuffled_order(example: Example, k: int, seed: int) -> tuple[int, ...]:
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
