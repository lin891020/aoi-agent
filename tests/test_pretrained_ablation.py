"""The ablation's arithmetic: what it pairs, and what it refuses to pair.

The measurement itself needs the dataset and a training run, so what is held
here is the part that can silently produce a wrong table without erroring: the
pairing. An ablation that compares one arm's median against the other's is
comparing across seeds, and the 2026-08-31 seed entry measured the same recipe
spanning 3.1 points with nothing changed -- wide enough to invent a difference
or hide one. So the delta a reader sees must be paired *within* a seed, and a
seed that lost one of its arms must drop out rather than pair with something
else.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ablation():
    pytest.importorskip("torch", reason="the ablation imports the training stack")
    return _script("pretrained_ablation")


def test_both_arms_are_named_so_a_row_says_which_one_it_is(ablation):
    assert ablation.ARMS == {True: "ImageNet", False: "random"}


def test_the_spread_is_median_and_range_not_mean_and_deviation(ablation):
    # Three seeds is a range, not a distribution. A mean with a standard
    # deviation over n=3 reads as though it were one.
    assert ablation.spread([0.50, 0.42, 0.34]) == {
        "median": 0.42, "min": 0.34, "max": 0.50, "n": 3}


def test_a_single_arm_is_summarised_without_dividing_by_zero(ablation):
    assert ablation.spread([0.5]) == {"median": 0.5, "min": 0.5, "max": 0.5, "n": 1}


def _rows(*triples):
    """(seed, arm, review removed) -> the row shape `pair_by_seed` reads."""
    return [{"seed": s, "arm": a, "oracle_review_reduction": r} for s, a, r in triples]


def test_the_delta_is_taken_within_a_seed_not_across_arms_medians(ablation):
    # Arm medians are 0.50 and 0.42, a 8-point difference. Paired, seed 0 gives
    # 16 points and seed 1 gives zero -- the medians describe neither.
    rows = _rows((0, "ImageNet", 0.58), (0, "random", 0.42),
                 (1, "ImageNet", 0.42), (1, "random", 0.42))
    assert ablation.pair_by_seed(rows, [0, 1]) == [
        {"seed": 0, "delta_review_removed": pytest.approx(0.16)},
        {"seed": 1, "delta_review_removed": pytest.approx(0.0)},
    ]


def test_a_seed_missing_an_arm_drops_out_rather_than_pairing_with_a_neighbour(ablation):
    rows = _rows((0, "ImageNet", 0.52), (0, "random", 0.42), (1, "ImageNet", 0.50))
    assert [p["seed"] for p in ablation.pair_by_seed(rows, [0, 1])] == [0]


def test_an_arm_that_reached_no_point_inside_the_budget_is_not_a_zero(ablation):
    # `None` is "this arm never got inside the budget", which is not 0% removed.
    rows = [{"seed": 0, "arm": "ImageNet", "oracle_review_reduction": 0.52},
            {"seed": 0, "arm": "random", "oracle_review_reduction": None}]
    assert ablation.pair_by_seed(rows, [0]) == []
