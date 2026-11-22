"""Тесты частей без модели: награды, промпт, разбиение, advantage, метрики."""

from __future__ import annotations

import numpy as np
import pytest

from rl_judge.config import RewardConfig, load_config
from rl_judge.data import Candidate, Example, split_examples
from rl_judge.evaluate import metrics, predictions
from rl_judge.grpo import group_advantages, sample_actions
from rl_judge.prompt import ABSTAIN, build_prompts, shuffled_order
from rl_judge.rewards import branch_share, reward, shared_depth

GOLD = "1/10-230-60-5/100"


def make_example(i: int = 0, gold_rank: int | None = 0, n: int = 5) -> Example:
    codes = [GOLD, "1/10-230-60-5/81", "1/10-230-60-7/3", "7/8-80/60", "2/1-1/1", "3/3/3"][:n]
    if gold_rank is None:
        codes[0] = "9/9/9"
    elif gold_rank > 0:
        codes[0], codes[gold_rank] = codes[gold_rank], codes[0]
    return Example(
        id=f"id{i}",
        query=f"запрос {i}",
        unit="т",
        gold=GOLD,
        slice="org_1_4",
        gold_rank=gold_rank,
        candidates=tuple(Candidate(code=c, name=f"имя {c}", unit="т") for c in codes),
    )


def test_shared_depth_and_branch() -> None:
    assert shared_depth(GOLD, "1/10-230-60-5/81") == 5
    assert branch_share("1/10-230-60-5/81", GOLD) == 1.0  # сосед по ветке
    assert branch_share("1/10-230-60-7/3", GOLD) == pytest.approx(4 / 5)
    assert branch_share("7/8-80/60", GOLD) == 0.0
    assert shared_depth("1/10", "1/100") == 1  # отрезки сравниваются целиком, не по символам


def test_rewards() -> None:
    binary = RewardConfig(name="binary")
    hier = RewardConfig(name="hierarchical", partial_credit=0.5)
    abstain = RewardConfig(name="abstain", wrong_penalty=2.0)
    for cfg in (binary, hier, abstain):
        assert reward(GOLD, GOLD, cfg) == 1.0
        assert reward(None, GOLD, cfg) == 0.0
    assert reward("1/10-230-60-5/81", GOLD, binary) == 0.0
    assert reward("1/10-230-60-5/81", GOLD, hier) == 0.5
    assert reward("1/10-230-60-5/81", GOLD, abstain) == -2.0


def test_prompt_order_is_fixed_and_maps_back() -> None:
    ex = make_example(gold_rank=2)
    assert shuffled_order(ex, 5, 42) == shuffled_order(ex, 5, 42)
    [prompt] = build_prompts([ex], k=5, seed=42)
    gold_action = prompt.gold_action()
    assert prompt.code_of(gold_action) == GOLD
    assert prompt.code_of(ABSTAIN) is None
    assert f"{gold_action}. имя {GOLD}" in prompt.text
    assert "0. Ни один вариант не подходит" in prompt.text


def test_unsolvable_gold_action_is_abstain() -> None:
    [prompt] = build_prompts([make_example(gold_rank=None)], k=5, seed=42)
    assert prompt.gold_action() == ABSTAIN


def test_split_is_deterministic_and_disjoint() -> None:
    examples = [make_example(i) for i in range(50)]
    a = split_examples(examples, dev_size=5, eval_size=10, seed=1)
    b = split_examples(list(reversed(examples)), dev_size=5, eval_size=10, seed=1)
    assert [e.id for e in a.eval] == [e.id for e in b.eval]
    ids = [{e.id for e in part} for part in (a.train, a.dev, a.eval)]
    assert sum(len(s) for s in ids) == 50
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_group_advantages() -> None:
    adv = group_advantages(np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    assert adv[0].tolist() == [1.0, -1.0, -1.0, 1.0]
    assert adv[1].tolist() == [0.0, 0.0, 0.0, 0.0]  # нет разброса, нет сигнала


def test_sample_actions_follows_distribution() -> None:
    logp = np.log(np.array([[0.0001, 0.9999, 0.0], [0.5, 0.5, 0.0]], dtype=np.float32) + 1e-12)
    actions = sample_actions(logp, group_size=1000, temperature=1.0, rng=np.random.default_rng(0))
    assert (actions[0] == 1).mean() > 0.99
    assert 0.4 < (actions[1] == 0).mean() < 0.6
    assert not (actions == 2).any()


def test_metrics_on_handmade_predictions() -> None:
    examples = [
        make_example(0, gold_rank=0),
        make_example(1, gold_rank=3),
        make_example(2, gold_rank=None),
    ]
    prompts = build_prompts(examples, k=5, seed=42)
    logp = np.full((3, 6), -10.0, dtype=np.float32)
    logp[0, prompts[0].gold_action()] = 0.0  # верно
    logp[1, 1 if prompts[1].gold_action() != 1 else 2] = 0.0  # неверно
    logp[2, ABSTAIN] = 0.0  # верного нет, воздержалась
    mass = np.ones(3, dtype=np.float32)
    m = metrics(predictions(prompts, logp, mass, k=5))
    assert m["n_solvable"] == 2
    assert m["accuracy"] == 0.5
    assert m["catboost_top1"] == 0.5
    assert m["abstain_when_unsolvable"] == 1.0
    assert m["coverage"] == pytest.approx(2 / 3, abs=1e-4)
    assert sum(m["answer_share"].values()) == pytest.approx(1.0, abs=1e-3)  # доли округлены
    assert m["errors"]["n"] == 1
    assert m["errors"]["same_branch"] + m["errors"]["other_branch"] == 1.0


def test_default_config_loads_with_override() -> None:
    cfg = load_config(["config/default.yaml"], ["reward.name=abstain", "train.seed=3"])
    assert cfg.reward.name == "abstain" and cfg.train.seed == 3
    with pytest.raises(ValueError):
        load_config(["config/default.yaml"], ["train.stepz=1"])
