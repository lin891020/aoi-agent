"""The train/val split, and the leak it exists to prevent.

Patches are crops from a board. A single DeepPCB image yields tens of them,
sharing its lighting, its registration error and often the same defect seen
from two candidate boxes. Split those patches at random and the validation set
is not held out from anything: for most val patches, a near-duplicate sat in
the training batch. Validation accuracy goes up, the operating-point curve
moves left, the escape rate at a given threshold falls -- and every one of
those numbers is wrong in the flattering direction.

That is the failure this file exists for, and until 2026-08-23 nothing in the
suite would have caught it. `split_by_image` had a docstring saying what it
must do and no test saying so. Replacing its body with a patch-level shuffle
left all 691 tests green, on a project whose every published figure is read off
this split.

The second test is the one that makes the first mean something. A leak test
that only ever sees a correct split cannot tell you it would notice a broken
one -- it passes just as happily against an assertion that can never fire. So
the same assertion is run against a deliberately leaking split, and it has to
fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from train import split_by_image

from aoi_agent.vision.patches import PatchSet


def patch_set(patches_per_image: tuple[int, ...]) -> PatchSet:
    """A PatchSet with a known image for every patch.

    The pixels are never read here -- `split_by_image` only ever sees
    `image_index` -- so they are 1x1x1 and the whole fixture costs nothing.
    """
    image_index = np.repeat(
        np.arange(len(patches_per_image), dtype=np.int64),
        np.array(patches_per_image, dtype=np.int64),
    )
    total = len(image_index)
    return PatchSet(
        patches=np.zeros((total, 3, 1, 1), dtype=np.uint8),
        labels=np.zeros(total, dtype=np.int64),
        label_names=["false_call", "open"],
        image_index=image_index,
        boxes=np.zeros((total, 4), dtype=np.int64),
    )


def images_on_each_side(patch_set_, train_idx, val_idx):
    index = patch_set_.image_index
    return set(index[train_idx].tolist()), set(index[val_idx].tolist())


def split_by_patch(patch_set_: PatchSet, val_fraction: float, seed: int):
    """The wrong split, kept here so the leak test can be shown to work.

    This is what `split_by_image` would be if someone simplified it into the
    obvious thing: shuffle the rows, cut. It is the exact mutation that left
    the suite green before this file existed.
    """
    order = np.arange(len(patch_set_.image_index))
    np.random.default_rng(seed).shuffle(order)
    cut = int(len(order) * (1 - val_fraction))
    return list(order[:cut]), list(order[cut:])


# 40 boards carrying a realistically uneven number of patches each: some boards
# contribute one candidate, some contribute thirty. An even fixture would hide
# a split that cut on patch counts rather than on images.
UNEVEN = tuple((i % 7) * 5 + 1 for i in range(40))


@pytest.mark.parametrize("seed", [0, 1, 7, 2026])
def test_no_image_appears_on_both_sides_of_the_split(seed):
    """The invariant itself: no board's patches straddle the split."""
    subject = patch_set(UNEVEN)
    train_idx, val_idx = split_by_image(subject, val_fraction=0.15, seed=seed)

    trained_on, validated_on = images_on_each_side(subject, train_idx, val_idx)
    assert not (trained_on & validated_on), (
        f"images {sorted(trained_on & validated_on)} have patches in both the "
        "training and the validation set; validation is measuring memorisation"
    )


def test_a_patch_level_split_is_what_this_test_would_catch():
    """The assertion above, run against a split that leaks. It must fail.

    Without this, a `split_by_image` that returned two empty lists would pass
    the leak test and nothing would say so.
    """
    subject = patch_set(UNEVEN)
    train_idx, val_idx = split_by_patch(subject, val_fraction=0.15, seed=0)

    trained_on, validated_on = images_on_each_side(subject, train_idx, val_idx)
    assert trained_on & validated_on, (
        "a patch-level shuffle of 40 multi-patch boards did not put a single "
        "board on both sides -- the fixture is not exercising the leak, so the "
        "test above proves nothing"
    )


def test_every_patch_lands_on_exactly_one_side():
    """A split that quietly drops patches would satisfy the leak test too."""
    subject = patch_set(UNEVEN)
    train_idx, val_idx = split_by_image(subject, val_fraction=0.15, seed=0)

    assert sorted(train_idx + val_idx) == list(range(len(subject)))


def test_the_holdout_is_about_the_size_that_was_asked_for():
    """Holding out whole images cannot hit a fraction exactly, but it must not
    collapse: an empty validation set passes every assertion above."""
    subject = patch_set(UNEVEN)
    train_idx, val_idx = split_by_image(subject, val_fraction=0.15, seed=0)

    _, validated_on = images_on_each_side(subject, train_idx, val_idx)
    assert len(validated_on) == 6, "40 images at 15% is 6 held out"
    assert 0.05 < len(val_idx) / len(subject) < 0.35, (
        "the patch-level holdout is a long way from the image-level fraction; "
        "with uneven boards that is expected, but not by this much"
    )


def test_the_same_seed_splits_the_same_way():
    """A run has to be reproducible from its seed -- the checkpoint, the
    operating point and the escape rate are all read off one split."""
    subject = patch_set(UNEVEN)
    first = split_by_image(subject, val_fraction=0.15, seed=3)
    second = split_by_image(subject, val_fraction=0.15, seed=3)
    other = split_by_image(subject, val_fraction=0.15, seed=4)

    assert first == second
    assert first != other, "two seeds gave the same split; the seed is being ignored"
