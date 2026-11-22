"""Три награды. Считаются по кодам, модель не нужна.

Код справочника - путь по дереву: в ``1/10-230-60-5/100`` раздел 1, подраздел 10,
и так до листа 100. Общая часть пути двух кодов - общий префикс по отрезкам.

* binary: 1 за верный код, иначе 0
* hierarchical: 1 за верный, за неверный - кредит по общей части пути,
  сосед по ветке получает весь ``partial_credit``
* abstain: +1 за верный, 0 за "не знаю", -λ за ошибку. Отвечать выгодно, когда
  уверенность выше λ / (1 + λ), при λ=1 это 50%
"""

from __future__ import annotations

import re

from rl_judge.config import RewardConfig

_SEPARATORS = re.compile(r"[/-]")


def code_path(code: str) -> list[str]:
    return _SEPARATORS.split(code)


def shared_depth(a: str, b: str) -> int:
    """Сколько первых отрезков пути совпадает."""
    depth = 0
    for x, y in zip(code_path(a), code_path(b), strict=False):
        if x != y:
            break
        depth += 1
    return depth


def branch_share(chosen: str, gold: str) -> float:
    """Какую долю пути до родителя верного кода выбранный код с ним делит, 0..1."""
    ancestors = len(code_path(gold)) - 1
    if ancestors <= 0:
        return 0.0
    return min(shared_depth(chosen, gold), ancestors) / ancestors


def reward(chosen: str | None, gold: str, cfg: RewardConfig) -> float:
    """``chosen`` None, если модель ответила 0 (не знаю)."""
    if chosen == gold:
        return 1.0
    if chosen is None:
        return 0.0
    if cfg.name == "binary":
        return 0.0
    if cfg.name == "hierarchical":
        return cfg.partial_credit * branch_share(chosen, gold)
    return -cfg.wrong_penalty
