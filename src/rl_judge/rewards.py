"""Три награды, которые мы сравниваем. Все считаются по коду, без модели.

Код справочника устроен как путь по дереву: ``1/10-230-60-5/100`` — это раздел 1,
подраздел 10, группа 230 и так далее до листа 100. Отрезки разделены «/» или «-»,
поэтому общая часть пути двух кодов — это их общий префикс по отрезкам.

* binary — 1 за верный код, иначе 0 (и за «не знаю» тоже 0);
* hierarchical — 1 за верный код, а за неверный лист даётся частичный кредит
  пропорционально общей части пути. Соседний лист той же ветки получает весь
  ``partial_credit``. Как раз здесь ждём лазейку: модели может стать выгодно
  угадывать ветку, а не лист;
* abstain — +1 за верный код, 0 за «не знаю», −λ за неверный. Отвечать выгодно,
  только если модель уверена больше чем на λ / (1 + λ); при λ=1 это 50%.
"""

from __future__ import annotations

import re

from rl_judge.config import RewardConfig

_SEPARATORS = re.compile(r"[/-]")


def code_path(code: str) -> list[str]:
    return _SEPARATORS.split(code)


def shared_depth(a: str, b: str) -> int:
    """Сколько первых отрезков пути у двух кодов совпадает."""
    depth = 0
    for x, y in zip(code_path(a), code_path(b), strict=False):
        if x != y:
            break
        depth += 1
    return depth


def branch_share(chosen: str, gold: str) -> float:
    """Доля пути до родителя верного кода, которую выбранный код делит с ним (0..1)."""
    ancestors = len(code_path(gold)) - 1
    if ancestors <= 0:
        return 0.0
    return min(shared_depth(chosen, gold), ancestors) / ancestors


def reward(chosen: str | None, gold: str, cfg: RewardConfig) -> float:
    """``chosen`` — выбранный код, None — модель ответила «0, не знаю»."""
    if chosen == gold:
        return 1.0
    if chosen is None:
        return 0.0
    if cfg.name == "binary":
        return 0.0
    if cfg.name == "hierarchical":
        return cfg.partial_credit * branch_share(chosen, gold)
    return -cfg.wrong_penalty
