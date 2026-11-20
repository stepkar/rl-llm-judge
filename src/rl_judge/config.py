"""Конфиг проекта: всё, что меняет результат, живёт в `config/default.yaml`.

Любую настройку можно переопределить из командной строки, например
``--set reward.name=abstain --set grpo.seed=1``. Значение разбирается как YAML,
так что числа, булевы и строки работают без кавычек. Опечатка в ключе — ошибка,
а не молча проигнорированная настройка (так же устроено в ml_classification).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Strict):
    path: Path = Path("data/full_val.jsonl")
    k: int = Field(default=5, ge=2, le=9)  # кандидатов в промпте; варианты ответа — цифры 0..k
    eval_size: int = 2000  # отложенная выборка: только до и после обучения, по ней все выводы
    dev_size: int = 500  # оценки по ходу обучения, кривые, подбор lr / kl_coef
    split_seed: int = 42  # фиксирует и разбиение, и порядок кандидатов в промптах


class ModelConfig(_Strict):
    name: str = "Qwen/Qwen3-4B-Instruct-2507"  # репозиторий HF или локальный путь
    lora_rank: int = 16
    lora_scale: float = 20.0  # множитель LoRA-добавки (так называется в mlx-lm, не alpha)
    cache_limit_gb: float = 8.0  # без лимита MLX копит освобождённые буферы до 100+ ГБ
    lora_layers: int = -1  # сколько последних блоков получают LoRA; -1 = все
    batch_size: int = 16  # батч инференса (оценка, референс); на обучение не влияет


class RewardConfig(_Strict):
    name: Literal["binary", "hierarchical", "abstain"] = "binary"
    partial_credit: float = 0.5  # hierarchical: максимум за неверный лист в верной ветке
    wrong_penalty: float = 1.0  # abstain: штраф λ за неверный ответ


class TrainConfig(_Strict):
    method: Literal["grpo", "sft"] = "grpo"
    steps: int = 250
    batch_size: int = 8  # промптов на шаг
    group_size: int = 8  # G: ответов на промпт (только grpo)
    # Ответ — один токен, и Adam двигает все 10 млн параметров LoRA разом: уже 1e-6
    # даёт KL ≈ 0.15 за один шаг на Qwen3-0.6B, 2e-5 ломает модель сразу. См. README.
    learning_rate: float = 3e-7
    kl_coef: float = 0.05
    temperature: float = 1.0  # температура сэмплирования группы
    eval_every: int = 50  # шагов между оценками на dev (оценка на 4B ≈ 30 с, ~10 шагов)
    seed: int = 0  # seed обучения: инициализация LoRA, порядок батчей, сэмплирование


class Config(_Strict):
    experiment: str = "debug"
    out_dir: Path = Path("runs")
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    reward: RewardConfig = RewardConfig()
    train: TrainConfig = TrainConfig()


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _apply_override(raw: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"--set ждёт KEY=VALUE, получено: {override!r}")
    dotted, value = override.split("=", 1)
    *parents, leaf = dotted.strip().split(".")
    node = raw
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = yaml.safe_load(value)


def load_config(paths: Sequence[str | Path], overrides: Sequence[str] = ()) -> Config:
    """Читает YAML по порядку (поздние перекрывают ранние) и применяет ``--set``."""
    raw: dict[str, Any] = {}
    for path in paths:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        raw = _deep_merge(raw, loaded)
    for override in overrides:
        _apply_override(raw, override)
    return Config.model_validate(raw)


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        metavar="YAML",
        help="конфиг (по умолчанию config/default.yaml); можно несколько",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="переопределить настройку, например --set reward.name=abstain",
    )


def config_from_args(args: argparse.Namespace) -> Config:
    return load_config(args.config or ["config/default.yaml"], args.overrides)


def dump_config(cfg: Config, path: Path) -> None:
    path.write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
