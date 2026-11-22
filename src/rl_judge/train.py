"""Обучение:grpo с выбранной наградой/sft.

    uv run python -m rl_judge.train --set experiment=abstain_s1 --set reward.name=abstain --set train.seed=1
    uv run python -m rl_judge.train --set experiment=sft_s0 --set train.method=sft

Батчи расписаны заранее, и референс для KL - только по промптам, которые
в них попадут (~2к из 10). Дальше исходная модель оценивается на dev и на
отложенной выборке (шаг 0), вставляется LoRA и идёт обучение с оценкой на dev
каждые train.eval_every шагов. В конце снова отложенная выборка, берётся последний шаг.

Что остаётся в runs/<время>_<experiment>/:
    config.yaml       конфиг прогона
    metrics.jsonl     kind=train на каждый шаг, kind=eval на каждую оценку на dev
    eval_step0.json   базовая модель на отложенной выборке (+ .predictions.jsonl)
    eval_final.json   обученная модель там же (+ .predictions.jsonl)
    adapters/         LoRA в формате mlx-lm
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.nn.utils import value_and_grad
from numpy.typing import NDArray

from rl_judge.config import Config, add_config_args, config_from_args, dump_config
from rl_judge.data import load_examples, split_examples
from rl_judge.evaluate import metrics, predictions, save_predictions
from rl_judge.grpo import (
    entropy,
    group_advantages,
    grpo_loss,
    mean_kl,
    sample_actions,
    sft_loss,
)
from rl_judge.model import (
    Judge,
    add_lora,
    encode_prompts,
    load_judge,
    option_logprobs,
    predict_logprobs,
    save_adapters,
    trainable_share,
)
from rl_judge.progress import Eta, fmt_duration
from rl_judge.prompt import ABSTAIN, Prompt, build_prompts
from rl_judge.rewards import reward

log = logging.getLogger("rl_judge.train")
HOLDOUT = "отложенная"


class MetricsLog:
    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, row: dict[str, Any]) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def reward_table(prompts: list[Prompt], cfg: Config) -> NDArray[np.float32]:
    """Награда за каждую цифру каждого промпта, [N, k+1]."""
    return np.array(
        [
            [reward(p.code_of(a), p.example.gold, cfg.reward) for a in range(p.n_options)]
            for p in prompts
        ],
        dtype=np.float32,
    )


def batch_schedule(
    n: int, steps: int, batch_size: int, rng: np.random.Generator
) -> list[NDArray[np.int64]]:
    """Индексы промптов на каждый шаг, на новой эпохе новое перемешивание."""
    batches = []
    order, cursor = rng.permutation(n), 0
    for _ in range(steps):
        if cursor + batch_size > n:
            order, cursor = rng.permutation(n), 0
        batches.append(order[cursor : cursor + batch_size])
        cursor += batch_size
    return batches


def evaluate_now(
    judge: Judge,
    prompts: list[Prompt],
    encoded: list[list[int]],
    cfg: Config,
    label: str,
    out: Path | None = None,
    ref: NDArray[np.float32] | None = None,
) -> tuple[dict[str, Any], NDArray[np.float32]]:
    """Метрики и распределения. С ``ref`` (исходная модель на тех же задачах) ещё и KL:
    на батче из 8 промптов он шумит, а на фиксированной выборке сравним между шагами."""
    logp, mass = predict_logprobs(judge, encoded, cfg.model.batch_size, label=label)
    preds = predictions(prompts, logp, mass, cfg.data.k)
    report = metrics(preds)
    report["kl"] = round(mean_kl(logp, ref), 4) if ref is not None else 0.0
    if out is not None:
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        save_predictions(preds, out.with_suffix(".predictions.jsonl"))
    return report, logp


def _eval_line(label: str, step: int, report: dict[str, Any]) -> str:
    return (
        f"{label}, шаг {step}: accuracy {report['accuracy']:.3f} "
        f"(CatBoost {report['catboost_top1']:.3f}), та же ветка {report['same_branch']:.3f}, "
        f'"не знаю" {report["abstain_rate"]:.1%}, '
        f"точность ответов {report['precision_answered']:.3f}, "
        f"энтропия {report['entropy']:.2f}, KL {report['kl']:.3f}"
    )


def train(cfg: Config, run_dir: Path) -> None:
    t = cfg.train
    d = cfg.data
    mx.random.seed(t.seed)
    rng = np.random.default_rng(t.seed)
    history = MetricsLog(run_dir / "metrics.jsonl")

    splits = split_examples(
        load_examples(d.path), dev_size=d.dev_size, eval_size=d.eval_size, seed=d.split_seed
    )
    train_prompts = build_prompts(splits.train, d.k, d.split_seed)
    dev_prompts = build_prompts(splits.dev, d.k, d.split_seed)
    eval_prompts = build_prompts(splits.eval, d.k, d.split_seed)
    log.info(
        "задачи: train %d, dev %d, отложенная %d (решаемых %.0f%% / %.0f%% / %.0f%%), k=%d",
        len(train_prompts),
        len(dev_prompts),
        len(eval_prompts),
        *(
            100 * np.mean([p.example.solvable(d.k) for p in prompts])
            for prompts in (train_prompts, dev_prompts, eval_prompts)
        ),
        d.k,
    )

    judge = load_judge(cfg.model, d.k)
    enc_train = encode_prompts(judge, train_prompts)
    enc_dev = encode_prompts(judge, dev_prompts)
    enc_eval = encode_prompts(judge, eval_prompts)
    log.info(
        "модель %s, длина промпта ≈ %d токенов",
        cfg.model.name,
        int(np.mean([len(e) for e in enc_train])),
    )

    schedule = batch_schedule(len(train_prompts), t.steps, t.batch_size, rng)
    used = np.unique(np.concatenate(schedule))
    log.info(
        "в обучение попадёт %d промптов из %d, референс только по ним", len(used), len(enc_train)
    )
    ref_used, _ = predict_logprobs(
        judge, [enc_train[i] for i in used], cfg.model.batch_size, label="референс"
    )
    ref_logp = np.zeros((len(train_prompts), ref_used.shape[1]), dtype=np.float32)
    ref_logp[used] = ref_used
    rewards = reward_table(train_prompts, cfg)
    gold_actions = np.array([p.gold_action() for p in train_prompts])

    dev, dev_ref = evaluate_now(judge, dev_prompts, enc_dev, cfg, "dev")
    history.write({"kind": "eval", "step": 0, **dev})
    log.info(_eval_line("dev", 0, dev))
    base, eval_ref = evaluate_now(
        judge, eval_prompts, enc_eval, cfg, HOLDOUT, run_dir / "eval_step0.json"
    )
    log.info(_eval_line(HOLDOUT, 0, base))

    add_lora(judge, cfg.model)
    log.info("LoRA: обучается %s параметров", trainable_share(judge.model))
    optimizer = optim.AdamW(learning_rate=t.learning_rate)

    def grpo_fn(
        tokens: list[list[int]], ref: mx.array, actions: mx.array, adv: mx.array
    ) -> tuple[mx.array, mx.array]:
        logp, _ = option_logprobs(judge, tokens)
        return grpo_loss(logp, ref, actions, adv, t.kl_coef)

    def sft_fn(
        tokens: list[list[int]], ref: mx.array, targets: mx.array
    ) -> tuple[mx.array, mx.array]:
        logp, _ = option_logprobs(judge, tokens)
        return sft_loss(logp, ref, targets)

    grpo_step = value_and_grad(judge.model, grpo_fn)
    sft_step = value_and_grad(judge.model, sft_fn)

    eta = Eta(t.steps)
    signal_total = 0.0
    for step, idx in enumerate(schedule, start=1):
        tokens = [enc_train[i] for i in idx]
        ref = mx.array(ref_logp[idx])

        # отдельный проход без градиента: группу надо сэмплировать до шага
        logp_now, _ = option_logprobs(judge, tokens)
        logp_np = np.array(logp_now)
        row: dict[str, Any] = {"kind": "train", "step": step}
        if t.method == "grpo":
            actions = sample_actions(logp_np, t.group_size, t.temperature, rng)
            group_rewards = rewards[idx][np.arange(len(idx))[:, None], actions]
            adv = group_advantages(group_rewards)
            (loss, kl), grads = grpo_step(tokens, ref, mx.array(actions), mx.array(adv))
            row["reward"] = float(group_rewards.mean())
            row["groups_with_signal"] = float((np.abs(adv).sum(axis=1) > 0).mean())
            signal_total += row["groups_with_signal"]
            row["abstain_sampled"] = float((actions == ABSTAIN).mean())
        else:
            (loss, kl), grads = sft_step(tokens, ref, mx.array(gold_actions[idx]))
        optimizer.update(judge.model, grads)
        mx.eval(judge.model.trainable_parameters(), optimizer.state, loss, kl)

        greedy = logp_np.argmax(axis=1)
        row.update(
            loss=float(loss),
            kl=float(kl),
            reward_greedy=float(rewards[idx, greedy].mean()),
            accuracy_greedy=float((greedy == gold_actions[idx]).mean()),
            entropy=entropy(logp_np),
            completion_tokens=1,  # длина ответа всегда 1, задание просит её логировать
        )
        history.write(row)

        if step % t.eval_every == 0 or step == t.steps:
            dev, _ = evaluate_now(judge, dev_prompts, enc_dev, cfg, "dev", ref=dev_ref)
            history.write({"kind": "eval", "step": step, **dev})
            log.info(_eval_line("dev", step, dev))
        if eta.due() or step == t.steps:
            # группа с одинаковыми наградами даёт градиент только от KL
            signal = f" | групп с сигналом {signal_total / step:.0%}" if t.method == "grpo" else ""
            log.info(
                "шаг %s | reward %.3f | kl %.4f | энтропия %.2f%s",
                eta.line(step),
                row.get("reward", row["reward_greedy"]),
                row["kl"],
                row["entropy"],
                signal,
            )

    save_adapters(judge, cfg.model, run_dir / "adapters")
    final, _ = evaluate_now(
        judge, eval_prompts, enc_eval, cfg, HOLDOUT, run_dir / "eval_final.json", ref=eval_ref
    )
    log.info("готово за %s. %s", fmt_duration(eta.elapsed), _eval_line(HOLDOUT, t.steps, final))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_config_args(parser)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s | %(message)s", datefmt="%H:%M:%S"
    )
    cfg = config_from_args(args)
    run_dir = cfg.out_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{cfg.experiment}"
    run_dir.mkdir(parents=True)
    dump_config(cfg, run_dir / "config.yaml")
    log.info(
        "прогон %s: %s, награда %s, seed %d",
        run_dir,
        cfg.train.method,
        cfg.reward.name,
        cfg.train.seed,
    )
    train(cfg, run_dir)


if __name__ == "__main__":
    main()
