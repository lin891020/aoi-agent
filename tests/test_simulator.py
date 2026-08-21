import numpy as np
import pytest

from aoi_agent.aoi.simulator import (
    DetectorConfig,
    Perturbation,
    apply_perturbation,
    detect,
)


def test_identical_boards_produce_no_candidates(blank_board):
    assert detect(blank_board, blank_board.copy()) == []


def test_detects_an_added_defect(blank_board):
    test = blank_board.copy()
    test[60:75, 60:75] = 255  # spurious copper

    candidates = detect(blank_board, test)

    assert len(candidates) == 1
    box = candidates[0].box
    assert box[0] <= 60 and box[1] <= 60
    assert box[2] >= 75 and box[3] >= 75


def test_min_area_filters_speckle(blank_board):
    test = blank_board.copy()
    test[64, 64] = 255  # a single stray pixel

    assert detect(blank_board, test, DetectorConfig(min_area=25)) == []
    loose = detect(
        blank_board, test, DetectorConfig(min_area=1, open_kernel=0, dilate_kernel=0)
    )
    assert len(loose) == 1


def test_max_area_rejects_registration_failure(blank_board):
    test = np.full_like(blank_board, 255)  # nothing lines up at all

    config = DetectorConfig(max_area=1000)
    assert detect(blank_board, test, config) == []


def test_candidates_stay_inside_the_image(blank_board):
    test = blank_board.copy()
    test[0:10, 0:10] = 255  # defect flush against the corner

    for candidate in detect(blank_board, test):
        assert candidate.x1 >= 0 and candidate.y1 >= 0
        assert candidate.x2 <= blank_board.shape[1]
        assert candidate.y2 <= blank_board.shape[0]


def test_perturbation_is_a_no_op_when_disabled(blank_board):
    assert np.array_equal(apply_perturbation(blank_board, Perturbation()), blank_board)


def test_misregistration_shows_up_in_the_difference_image(blank_board):
    """A shifted template makes real trace edges differ from the board.

    This is the dominant source of false calls on a real line: the stage never
    repeats exactly, so every edge in the image disagrees slightly.
    """
    perturbation = Perturbation(max_shift_px=3, seed=0)  # shifts by (2, 1)
    shifted = apply_perturbation(blank_board, perturbation)

    differing = int((np.abs(shifted.astype(int) - blank_board.astype(int)) > 60).sum())
    assert differing > 100


def test_default_morphology_suppresses_misregistration_artefacts(blank_board):
    """The default opening kernel erases thin edge artefacts -- by design.

    Registration error produces one- and two-pixel-wide slivers along every
    trace edge. A 3x3 opening removes them, which is what keeps the candidate
    count workable. The consequence is that this detector is more tolerant of
    misregistration than a real AOI, so the false-call counts it produces are
    conservative.
    """
    perturbation = Perturbation(max_shift_px=3, seed=0)

    with_morphology = detect(
        blank_board, blank_board.copy(), DetectorConfig(), perturbation
    )
    without_morphology = detect(
        blank_board,
        blank_board.copy(),
        DetectorConfig(open_kernel=0, dilate_kernel=0),
        perturbation,
    )

    assert with_morphology == []
    assert len(without_morphology) > 0


def test_zero_shift_is_possible_and_harmless(blank_board):
    """The shift is drawn inclusive of zero, so some boards land perfectly."""
    perturbation = Perturbation(max_shift_px=3, seed=1)  # draws (0, 0)
    assert np.array_equal(apply_perturbation(blank_board, perturbation), blank_board)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_perturbation_is_reproducible(blank_board, seed):
    perturbation = Perturbation(max_shift_px=3, noise_sigma=5.0, seed=seed)
    first = apply_perturbation(blank_board, perturbation)
    second = apply_perturbation(blank_board, perturbation)
    assert np.array_equal(first, second)
