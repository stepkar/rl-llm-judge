"""Судья на MLX: загрузка, LoRA, вероятности цифр.

Текст не генерируем. Берём логиты следующего токена, из них цифры 0..k, и
перенормируем - это и есть политика. Жадный ответ - самая вероятная цифра.
``option_mass`` - сколько вероятности модель сама отдаёт цифрам.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from numpy.typing import NDArray

from rl_judge.config import ModelConfig
from rl_judge.progress import Eta
from rl_judge.prompt import Prompt

log = logging.getLogger("rl_judge.model")


@dataclass
class Judge:
    model: Any
    tokenizer: Any
    option_ids: mx.array  # id токенов "0".."k" именно в этом порядке


def load_judge(cfg: ModelConfig, k: int) -> Judge:
    mx.set_cache_limit(int(cfg.cache_limit_gb * 1e9))
    loaded = load(cfg.name)
    model, tokenizer = loaded[0], loaded[1]
    ids = []
    for digit in range(k + 1):
        tokens = tokenizer.encode(str(digit), add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(f"цифра {digit} у токенизатора {cfg.name} не один токен: {tokens}")
        ids.append(tokens[0])
    return Judge(model=model, tokenizer=tokenizer, option_ids=mx.array(ids))


def add_lora(judge: Judge, cfg: ModelConfig) -> None:
    """B-матрица LoRA стартует с нуля, так что до первого шага модель равна исходной."""
    judge.model.freeze()
    linear_to_lora_layers(judge.model, cfg.lora_layers, _lora_parameters(cfg))


def _lora_parameters(cfg: ModelConfig) -> dict[str, float]:
    return {"rank": cfg.lora_rank, "scale": cfg.lora_scale, "dropout": 0.0}


def save_adapters(judge: Judge, cfg: ModelConfig, folder: Path) -> None:
    """В формате mlx-lm, грузится через ``load(name, adapter_path=folder)``."""
    folder.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(judge.model.trainable_parameters()))
    mx.save_safetensors(str(folder / "adapters.safetensors"), weights)
    config = {
        "fine_tune_type": "lora",
        "num_layers": cfg.lora_layers,
        "lora_parameters": _lora_parameters(cfg),
    }
    (folder / "adapter_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_with_adapters(cfg: ModelConfig, k: int, adapters: Path) -> Judge:
    judge = load_judge(cfg, k)
    add_lora(judge, cfg)
    judge.model.load_weights(str(adapters / "adapters.safetensors"), strict=False)
    return judge


def encode_prompts(judge: Judge, prompts: list[Prompt]) -> list[list[int]]:
    """Токены с chat-шаблоном, рассуждения выключены."""
    encoded = []
    for prompt in prompts:
        text = judge.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt.text}],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )
        encoded.append(judge.tokenizer.encode(text, add_special_tokens=False))
    return encoded


def _pad(batch: list[list[int]], pad_id: int) -> tuple[mx.array, mx.array]:
    """Паддинг справа. Внимание причинное, на последний настоящий токен он не влияет."""
    width = max(len(t) for t in batch)
    tokens = np.full((len(batch), width), pad_id, dtype=np.int32)
    for i, t in enumerate(batch):
        tokens[i, : len(t)] = t
    return mx.array(tokens), mx.array([len(t) for t in batch])


def option_logprobs(judge: Judge, batch: list[list[int]]) -> tuple[mx.array, mx.array]:
    """Log-вероятности вариантов [B, k+1] и масса на цифрах [B].

    Выходной слой (словарь 152 тыс.) считаем только на последней позиции,
    иначе логиты батча занимают гигабайты.
    """
    pad_id = judge.tokenizer.pad_token_id or judge.tokenizer.eos_token_id
    tokens, lengths = _pad(batch, pad_id)
    # у мультимодальных Qwen3.5 текстовая часть лежит в language_model
    lm = getattr(judge.model, "language_model", judge.model)
    hidden = lm.model(tokens)
    last = hidden[mx.arange(len(batch)), lengths - 1]
    if lm.args.tie_word_embeddings:
        logits = lm.model.embed_tokens.as_linear(last)
    else:
        logits = lm.lm_head(last)
    logits = logits.astype(mx.float32)
    options = logits[:, judge.option_ids]
    options_lse = mx.logsumexp(options, axis=-1, keepdims=True)
    mass = mx.exp(options_lse[:, 0] - mx.logsumexp(logits, axis=-1))
    return options - options_lse, mass


def predict_logprobs(
    judge: Judge, encoded: list[list[int]], batch_size: int, label: str = "инференс"
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """``option_logprobs`` по всему списку батчами, в numpy."""
    logps, masses = [], []
    eta = Eta(len(encoded))
    for lo in range(0, len(encoded), batch_size):
        logp, mass = option_logprobs(judge, encoded[lo : lo + batch_size])
        mx.eval(logp, mass)
        logps.append(np.array(logp))
        masses.append(np.array(mass))
        done = min(lo + batch_size, len(encoded))
        if eta.due() or done == len(encoded):
            log.info("%s: %s", label, eta.line(done))
    return np.concatenate(logps).astype(np.float32), np.concatenate(masses).astype(np.float32)


def _n_params(tree: Any) -> int:
    leaves = cast(list[tuple[str, mx.array]], tree_flatten(tree))
    return sum(int(v.size) for _, v in leaves)


def trainable_share(model: Any) -> str:
    trainable = _n_params(model.trainable_parameters())
    total = _n_params(model.parameters())
    return f"{trainable / 1e6:.1f} млн из {total / 1e6:.0f} млн ({trainable / total:.2%})"
