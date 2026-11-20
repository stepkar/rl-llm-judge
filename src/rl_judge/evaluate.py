"""Оценка судьи жадным ответом на фиксированной выборке.

Главная метрика — accuracy на «решаемых» задачах, где верный код есть среди
показанных k кандидатов: только на них судья в принципе может угадать. Но
считаем и всё остальное, иначе не увидеть, что делает награда с воздержанием:
долю «не знаю», точность среди тех ответов, что модель всё-таки дала, и как
часто она воздерживается там, где верного кода в списке действительно нет.

Отдельно (без обучения, просто посмотреть базовую модель):

    uv run python -m rl_judge.evaluate --set model.name=Qwen/Qwen3-0.6B
    uv run python -m rl_judge.evaluate --adapters runs/<прогон>/adapters --split eval
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rl_judge.config import Config, add_config_args, config_from_args
from rl_judge.data import load_examples, split_examples
from rl_judge.prompt import ABSTAIN, Prompt, build_prompts
from rl_judge.rewards import branch_share

log = logging.getLogger("rl_judge.evaluate")


@dataclass(frozen=True, slots=True)
class Prediction:
    id: str
    slice: str
    solvable: bool
    gold_action: int
    action: int  # что ответила модель: 1..k или 0
    correct: bool
    catboost_correct: bool  # CatBoost поставил верный код первым — с ним и сравниваем судью
    gold_rank: int | None  # место верного кода у CatBoost (0 = top-1), None — его нет в пуле
    catboost_rank: int | None  # место выбранного кандидата у CatBoost (0 = его top-1)
    same_branch: bool  # выбранный код в той же ветке, что верный (включая сам верный)
    probs: list[float]  # распределение по 0..k
    option_mass: float


def predictions(
    prompts: list[Prompt], logp: NDArray[np.float32], mass: NDArray[np.float32], k: int
) -> list[Prediction]:
    out = []
    for prompt, row, m in zip(prompts, logp, mass, strict=True):
        action = int(np.argmax(row))
        chosen = prompt.code_of(action)
        gold = prompt.example.gold
        out.append(
            Prediction(
                id=prompt.example.id,
                slice=prompt.example.slice,
                solvable=prompt.example.solvable(k),
                gold_action=prompt.gold_action(),
                action=action,
                correct=chosen == gold,
                catboost_correct=prompt.example.gold_rank == 0,
                gold_rank=prompt.example.gold_rank,
                catboost_rank=None if action == ABSTAIN else prompt.order[action - 1],
                same_branch=chosen is not None
                and (chosen == gold or branch_share(chosen, gold) == 1.0),
                probs=[round(float(p), 4) for p in np.exp(row)],
                option_mass=round(float(m), 4),
            )
        )
    return out


def _share(flags: list[bool]) -> float:
    return round(sum(flags) / len(flags), 4) if flags else 0.0


def _breakdown(preds: list[Prediction], key: Callable[[Prediction], Any]) -> dict[str, Any]:
    """Accuracy судьи и CatBoost по группам решаемых задач."""
    groups: dict[Any, list[Prediction]] = defaultdict(list)
    for p in preds:
        groups[key(p)].append(p)
    return {
        str(name): {
            "n": len(group),
            "accuracy": _share([p.correct for p in group]),
            "catboost": _share([p.catboost_correct for p in group]),
        }
        for name, group in sorted(groups.items())
    }


def metrics(preds: list[Prediction]) -> dict[str, Any]:
    solvable = [p for p in preds if p.solvable]
    unsolvable = [p for p in preds if not p.solvable]
    answered = [p for p in preds if p.action != ABSTAIN]
    wrong = [p for p in solvable if not p.correct]
    entropy = [-sum(q * np.log(q) for q in p.probs if q > 0) for p in preds]
    n_options = len(preds[0].probs) if preds else 0
    return {
        "n": len(preds),
        "n_solvable": len(solvable),
        "accuracy": _share([p.correct for p in solvable]),  # главная метрика
        "catboost_top1": _share([p.catboost_correct for p in solvable]),
        "same_branch": _share([p.same_branch for p in solvable]),
        "abstain_rate": _share([p.action == ABSTAIN for p in preds]),
        "coverage": round(len(answered) / len(preds), 4) if preds else 0.0,
        "precision_answered": _share([p.correct for p in answered]),
        "abstain_when_unsolvable": _share([p.action == ABSTAIN for p in unsolvable]),
        "agree_with_catboost": _share([p.catboost_rank == 0 for p in answered]),
        "entropy": round(float(np.mean(entropy)), 4) if entropy else 0.0,
        "option_mass": round(float(np.mean([p.option_mass for p in preds])), 4) if preds else 0.0,
        "slices": _breakdown(solvable, lambda p: p.slice),
        # Кандидаты перемешаны, так что верный ответ равномерно стоит на местах 1..k.
        # Если accuracy сильно зависит от места, модель выбирает по позиции, а не по тексту.
        "by_gold_position": _breakdown(solvable, lambda p: p.gold_action),
        "answer_share": {str(a): _share([p.action == a for p in preds]) for a in range(n_options)},
        # Где судья полезен: CatBoost поставил верный код на 1-е место или ниже.
        "by_catboost_rank": _breakdown(solvable, lambda p: p.gold_rank),
        # Из чего состоят ошибки на решаемых задачах. Рост «соседа по ветке» при
        # иерархической награде — та лазейка, которую мы ищем.
        "errors": {
            "n": len(wrong),
            "same_branch": _share([p.same_branch and p.action != ABSTAIN for p in wrong]),
            "abstain": _share([p.action == ABSTAIN for p in wrong]),
            "other_branch": _share([not p.same_branch and p.action != ABSTAIN for p in wrong]),
        },
    }


def save_predictions(preds: list[Prediction], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")


def run(cfg: Config, adapters: Path | None, split: str, out: Path | None) -> dict[str, Any]:
    # MLX грузим только здесь: метрики выше считаются и без GPU (тесты, отчёт).
    from rl_judge.model import encode_prompts, load_judge, load_with_adapters, predict_logprobs

    d = cfg.data
    splits = split_examples(
        load_examples(d.path), dev_size=d.dev_size, eval_size=d.eval_size, seed=d.split_seed
    )
    prompts = build_prompts(splits.eval if split == "eval" else splits.dev, d.k, d.split_seed)
    judge = load_with_adapters(cfg.model, d.k, adapters) if adapters else load_judge(cfg.model, d.k)
    logp, mass = predict_logprobs(judge, encode_prompts(judge, prompts), cfg.model.batch_size)
    preds = predictions(prompts, logp, mass, d.k)
    report = {
        "model": cfg.model.name,
        "adapters": str(adapters) if adapters else None,
        "split": split,
        **metrics(preds),
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        save_predictions(preds, out.with_suffix(".predictions.jsonl"))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_config_args(parser)
    parser.add_argument("--adapters", type=Path, default=None, help="каталог с LoRA-адаптерами")
    parser.add_argument("--split", choices=["eval", "dev"], default="eval")
    parser.add_argument("--out", type=Path, default=None, help="куда сохранить метрики (json)")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s | %(message)s", datefmt="%H:%M:%S"
    )
    report = run(config_from_args(args), args.adapters, args.split, args.out)
    log.info("%s", json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
