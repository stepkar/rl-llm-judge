"""Графики и таблицы для README по готовым прогонам.

    uv run python -m rl_judge.report runs/*        # → docs/figures/*.png, docs/results.md

Прогоны группируются по условию (метод + награда), seed внутри условия
усредняются: на графиках линия — среднее, полоса — min..max по seed, в таблице
«среднее ± стандартное отклонение». Если seed один, полосы нет, и выводы из
маленькой разницы делать не стоит — об этом же пишет и задание.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from rl_judge.config import Config
from rl_judge.data import load_examples, split_examples
from rl_judge.prompt import ABSTAIN, Prompt, build_prompts

# Метрики итоговой таблицы (ключ в eval_final.json → подпись).
TABLE_METRICS = {
    "accuracy": "accuracy",
    "same_branch": "та же ветка",
    "abstain_rate": "«не знаю»",
    "precision_answered": "точность ответов",
    "abstain_when_unsolvable": "«не знаю» без верного",
    "agree_with_catboost": "согласие с CatBoost",
    "entropy": "энтропия",
    "kl": "KL к исходной",
}
SLICES = {
    "only_reference": "только эталон",
    "org_1_4": "1–4 примера",
    "org_5plus": "5+ примеров",
}
ERRORS = {"same_branch": "сосед по ветке", "other_branch": "другая ветка", "abstain": "«не знаю»"}


@dataclass
class Run:
    path: Path
    cfg: Config
    train: list[dict[str, Any]]
    evals: list[dict[str, Any]]  # dev по шагам
    base: dict[str, Any]  # отложенная выборка до обучения
    final: dict[str, Any]  # и после

    @property
    def condition(self) -> str:
        if self.cfg.train.method == "sft":
            return "SFT"
        return f"GRPO · {self.cfg.reward.name}"


def load_run(path: Path) -> Run:
    cfg = Config.model_validate(yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8")))
    rows = [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return Run(
        path=path,
        cfg=cfg,
        train=[r for r in rows if r["kind"] == "train"],
        evals=[r for r in rows if r["kind"] == "eval"],
        base=json.loads((path / "eval_step0.json").read_text(encoding="utf-8")),
        final=json.loads((path / "eval_final.json").read_text(encoding="utf-8")),
    )


def _smooth(values: list[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    head = np.array([arr[: i + 1].mean() for i in range(window - 1)])
    return np.concatenate([head, np.convolve(arr, kernel, mode="valid")])


def _plot_band(ax: Any, steps: list[int], series: list[np.ndarray], label: str) -> None:
    stacked = np.vstack([s[: len(steps)] for s in series])
    line = ax.plot(steps, stacked.mean(axis=0), label=f"{label} (seed: {len(series)})")
    if len(series) > 1:
        ax.fill_between(
            steps, stacked.min(axis=0), stacked.max(axis=0), alpha=0.2, color=line[0].get_color()
        )


def plot_curves(groups: dict[str, list[Run]], out: Path, smooth: int = 10) -> list[Path]:
    """Кривые обучения: награда и KL на батчах, точность, «не знаю» и энтропия на dev."""
    figures = []
    specs = [
        (
            "train",
            "reward_greedy",
            "Награда жадного ответа на батче (скользящее среднее, 10 шагов)",
            "награда",
        ),
        ("train", "kl", "KL(π ‖ π_исх) на батче (скользящее среднее, 10 шагов)", "KL, наты"),
        ("eval", "kl", "KL(π ‖ π_исх) на dev", "KL, наты"),
        ("eval", "accuracy", "Accuracy на dev (решаемые задачи)", "доля верных"),
        ("eval", "abstain_rate", "Доля ответов «не знаю» на dev", "доля"),
        ("eval", "entropy", "Энтропия политики на dev", "энтропия, наты"),
    ]
    for kind, key, title, ylabel in specs:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for name, runs in sorted(groups.items()):
            rows = [r.train if kind == "train" else r.evals for r in runs]
            steps = [row["step"] for row in min(rows, key=len)]  # общая часть, если длины разные
            raw = [[row[key] for row in rs] for rs in rows]
            series = [_smooth(v, smooth) if kind == "train" else np.asarray(v) for v in raw]
            _plot_band(ax, steps, series, name)
        if key == "accuracy":
            catboost = next(iter(groups.values()))[0].evals[0]["catboost_top1"]
            ax.axhline(
                catboost, color="grey", linestyle="--", label=f"CatBoost top-1 ({catboost:.3f})"
            )
        ax.set_title(title)
        ax.set_xlabel("шаг обучения")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
        path = out / f"{kind}_{key}.png"  # train_kl и eval_kl не перезаписывают друг друга
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        figures.append(path)
    return figures


def _mean_std(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}"


def results_table(groups: dict[str, list[Run]]) -> str:
    """Итог по условиям: базовая модель (шаг 0) и каждое обученное условие."""
    first = next(iter(groups.values()))[0]
    base = first.base
    header = "| условие | seed | " + " | ".join(TABLE_METRICS.values()) + " |"
    lines = [header, "|" + "---|" * (len(TABLE_METRICS) + 2)]
    lines.append(
        "| базовая модель | — | " + " | ".join(f"{base[k]:.3f}" for k in TABLE_METRICS) + " |"
    )
    lines.append(
        f"| CatBoost top-1 | — | {first.final['catboost_top1']:.3f} | "
        + " | ".join("—" for _ in list(TABLE_METRICS)[1:])
        + " |"
    )
    for name, runs in sorted(groups.items()):
        cells = [_mean_std([r.final[k] for r in runs]) for k in TABLE_METRICS]
        lines.append(f"| {name} | {len(runs)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def breakdown_table(groups: dict[str, list[Run]], key: str, columns: dict[str, str]) -> str:
    """Accuracy по группам задач (срез, место верного ответа, место у CatBoost):
    строка CatBoost, базовая модель и каждое условие."""
    first = next(iter(groups.values()))[0]
    lines = [
        "| условие | " + " | ".join(columns.values()) + " |",
        "|" + "---|" * (len(columns) + 1),
        "| n задач | " + " | ".join(str(first.final[key][c]["n"]) for c in columns) + " |",
        "| CatBoost top-1 | "
        + " | ".join(f"{first.final[key][c]['catboost']:.3f}" for c in columns)
        + " |",
        "| базовая модель | "
        + " | ".join(f"{first.base[key][c]['accuracy']:.3f}" for c in columns)
        + " |",
    ]
    for name, runs in sorted(groups.items()):
        cells = [_mean_std([r.final[key][c]["accuracy"] for r in runs]) for c in columns]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def answer_share_table(groups: dict[str, list[Run]]) -> str:
    """Как часто модель называет каждую цифру. Верный ответ равномерно стоит на местах
    1..k, так что перекос в одну цифру — выбор по позиции, а не по тексту."""
    first = next(iter(groups.values()))[0]
    options = list(first.base["answer_share"])
    lines = [
        "| условие | " + " | ".join(f"«{o}»" for o in options) + " |",
        "|" + "---|" * (len(options) + 1),
        "| базовая модель | "
        + " | ".join(f"{first.base['answer_share'][o]:.3f}" for o in options)
        + " |",
    ]
    for name, runs in sorted(groups.items()):
        cells = [_mean_std([r.final["answer_share"][o] for r in runs]) for o in options]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def errors_table(groups: dict[str, list[Run]]) -> str:
    """Из чего состоят ошибки на решаемых задачах (доли от числа ошибок)."""
    first = next(iter(groups.values()))[0]
    lines = [
        "| условие | ошибок | " + " | ".join(ERRORS.values()) + " |",
        "|" + "---|" * (len(ERRORS) + 2),
        f"| базовая модель | {first.base['errors']['n']} | "
        + " | ".join(f"{first.base['errors'][e]:.3f}" for e in ERRORS)
        + " |",
    ]
    for name, runs in sorted(groups.items()):
        n = f"{np.mean([r.final['errors']['n'] for r in runs]):.0f}"
        cells = [_mean_std([r.final["errors"][e] for r in runs]) for e in ERRORS]
        lines.append(f"| {name} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _answer(prompt: Prompt, action: int) -> str:
    if action == ABSTAIN:
        return "0 (не знаю)"
    mark = "✓" if prompt.code_of(action) == prompt.example.gold else "✗"
    name = prompt.example.candidates[prompt.order[action - 1]].name
    return f"{mark} {name[:60]}"


def _holdout_prompts(run: Run) -> dict[str, Prompt]:
    d = run.cfg.data
    splits = split_examples(
        load_examples(d.path), dev_size=d.dev_size, eval_size=d.eval_size, seed=d.split_seed
    )
    return {p.example.id: p for p in build_prompts(splits.eval, d.k, d.split_seed)}


def _actions(path: Path) -> dict[str, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {row["id"]: row["action"] for row in map(json.loads, lines)}


def _gold_name(prompt: Prompt) -> str:
    return next(c.name for c in prompt.example.candidates if c.code == prompt.example.gold)


def error_examples(groups: dict[str, list[Run]], n: int) -> str:
    """Ошибки каждого условия (первый seed) на решаемых задачах: что выбрала модель
    вместо верного и в той ли это ветке справочника."""
    first = next(iter(groups.values()))[0]
    prompts = _holdout_prompts(first)
    lines = [
        "| условие | позиция | верный класс | ответ модели | ветка |",
        "|---|---|---|---|---|",
    ]
    for name, runs in sorted(groups.items()):
        path = runs[0].path / "eval_final.predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        wrong = sorted(
            (r for r in rows if r["solvable"] and not r["correct"]), key=lambda r: r["id"]
        )
        for r in wrong[:n]:
            p = prompts[r["id"]]
            branch = "—" if r["action"] == ABSTAIN else ("та же" if r["same_branch"] else "другая")
            lines.append(
                f"| {name} | {p.example.query[:50]} | {_gold_name(p)[:50]} | "
                f"{_answer(p, r['action'])} | {branch} |"
            )
    return "\n".join(lines)


def answers_table(groups: dict[str, list[Run]], n: int) -> str:
    """Ответы базовой и обученных моделей (первый seed каждого условия) на одних задачах."""
    first = next(iter(groups.values()))[0]
    d = first.cfg.data
    prompts = _holdout_prompts(first)
    base = _actions(first.path / "eval_step0.predictions.jsonl")
    trained = {
        name: _actions(runs[0].path / "eval_final.predictions.jsonl")
        for name, runs in sorted(groups.items())
    }
    names = list(trained)
    lines = [
        "| позиция | верный класс | базовая | " + " | ".join(names) + " |",
        "|" + "---|" * (len(names) + 3),
    ]
    # Сначала задачи, где ответы разошлись: на них видно, что поменяло обучение.
    ids = [i for i in base if prompts[i].example.solvable(d.k)]
    ids.sort(key=lambda i: len({base[i], *(trained[m][i] for m in names)}) == 1)
    for i in ids[:n]:
        p = prompts[i]
        gold = _gold_name(p)
        cells = [_answer(p, base[i])] + [_answer(p, trained[m][i]) for m in names]
        lines.append(f"| {p.example.query[:50]} | {gold[:50]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("runs", nargs="+", type=Path, help="каталоги прогонов из runs/")
    parser.add_argument("--out", type=Path, default=Path("docs"))
    parser.add_argument("--answers", type=int, default=15, help="строк в таблице ответов")
    parser.add_argument("--errors", type=int, default=5, help="примеров ошибок на условие")
    args = parser.parse_args()

    groups: dict[str, list[Run]] = defaultdict(list)
    for path in args.runs:
        if (path / "eval_final.json").exists():  # недоделанные прогоны пропускаем
            run = load_run(path)
            groups[run.condition].append(run)
    if not groups:
        raise SystemExit("ни одного завершённого прогона")

    figures_dir = args.out / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures = plot_curves(groups, figures_dir)
    first = next(iter(groups.values()))[0]
    positions = {str(a): f"на месте {a}" for a in range(1, first.cfg.data.k + 1)}
    ranks = {str(r): f"у CatBoost {r + 1}-й" for r in range(first.cfg.data.k)}
    text = [
        "# Результаты",
        "",
        "Все цифры — отложенная выборка, жадный ответ, последний шаг обучения. Accuracy и",
        "доли по ветке считаются на решаемых задачах (верный код есть среди показанных),",
        "«не знаю» и точность ответов — на всех. Для обученных условий «среднее ± стандартное",
        "отклонение по seed» (ddof=1), в столбце seed — число запусков. Базовая модель и",
        "CatBoost от seed не зависят. На графиках линия — среднее по seed, полоса — min..max.",
        "",
        "## Итоговое сравнение",
        "",
        results_table(groups),
        "",
        "## По срезам: сколько примеров у верного класса в train каскада",
        "",
        breakdown_table(groups, "slices", SLICES),
        "",
        "## По месту верного кода у CatBoost",
        "",
        "Первый столбец — задачи, где CatBoost и так прав; остальные — где судья может его поправить.",
        "",
        breakdown_table(groups, "by_catboost_rank", ranks),
        "",
        "## По месту верного ответа в промпте",
        "",
        "Кандидаты перемешаны, верный ответ стоит на каждом месте примерно одинаково часто.",
        "Если accuracy сильно зависит от места, модель выбирает по позиции, а не по тексту.",
        "",
        breakdown_table(groups, "by_gold_position", positions),
        "",
        "Как часто модель называет каждую цифру (все задачи, «0» — «не знаю»):",
        "",
        answer_share_table(groups),
        "",
        "## Из чего состоят ошибки",
        "",
        errors_table(groups),
        "",
        error_examples(groups, args.errors),
        "",
        "## Ответы моделей на одних и тех же задачах",
        "",
        answers_table(groups, args.answers),
        "",
        "## Графики",
        "",
        "Длина ответа у всех условий постоянна — одна цифра (`completion_tokens: 1` в",
        "metrics.jsonl), поэтому вместо неё показаны энтропия и доля «не знаю».",
        "",
        *[f"![{f.stem}](figures/{f.name})" for f in figures],
    ]
    (args.out / "results.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"готово: {args.out / 'results.md'}, графиков: {len(figures)}")


if __name__ == "__main__":
    main()
