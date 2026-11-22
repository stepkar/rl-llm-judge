"""Таски для судьи: чтение jsonl из ml_classification+фиксированное разбиение.

Строка jsonl — 1 позиция из заявки организации (после номрализации)+ верный код и до 10 кандидатов
после CatBoost (по убыванию его скора).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Candidate:
    code: str
    name: str
    unit: str | None


@dataclass(frozen=True, slots=True)
class Example:
    id: str
    query: str
    unit: str | None
    gold: str
    slice: str  # only_reference / org_1_4 / org_5plus: сколько примеров у класса в train каскада
    gold_rank: int | None  # место верного кода у CatBoost с нуля, None если его нет в пуле
    candidates: tuple[Candidate, ...]

    def solvable(self, k: int) -> bool:
        """Верный код есть среди первых k кандидатов."""
        return self.gold_rank is not None and self.gold_rank < k


@dataclass(frozen=True)
class Splits:
    train: list[Example]
    dev: list[Example]
    eval: list[Example]


def load_examples(path: Path) -> list[Example]:
    examples = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            examples.append(
                Example(
                    id=row["id"],
                    query=row["query"],
                    unit=row["unit"],
                    gold=row["gold"],
                    slice=row["slice"],
                    gold_rank=row["gold_rank"],
                    candidates=tuple(
                        Candidate(code=c["code"], name=c["name"], unit=c["unit"])
                        for c in row["candidates"]
                    ),
                )
            )
    return examples


def _bucket_key(example_id: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{example_id}".encode()).hexdigest()


def split_examples(examples: list[Example], *, dev_size: int, eval_size: int, seed: int) -> Splits:
    """Сортируем по хэшу (seed, id) и режем. От порядка строк в файле не зависит.

    id это хэш текста. Около 12% строк - один текст с разными кодами, и они всегда
    попадают в одну часть, так что одинаковый промпт не бывает и в train, и в eval.
    """
    if dev_size + eval_size >= len(examples):
        raise ValueError(
            f"dev_size + eval_size = {dev_size + eval_size}, а задач всего {len(examples)}"
        )
    ordered = sorted(examples, key=lambda e: _bucket_key(e.id, seed))
    return Splits(
        eval=ordered[:eval_size],
        dev=ordered[eval_size : eval_size + dev_size],
        train=ordered[eval_size + dev_size :],
    )
