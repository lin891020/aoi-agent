import numpy as np

from aoi_agent.vision.patches import PATCH_SIZE, PatchSet, build_patch, crop

from conftest import make_candidate


def test_crop_returns_the_requested_size(blank_board):
    patch = crop(blank_board, make_candidate(60, 60, 70, 70), size=32)
    assert patch.shape == (32, 32)


def test_crop_pads_at_the_image_edge(blank_board):
    """A defect in the corner still yields a full-size window."""
    patch = crop(blank_board, make_candidate(0, 0, 8, 8), size=32)
    assert patch.shape == (32, 32)


def test_crop_is_centred_on_the_candidate():
    board = np.zeros((128, 128), dtype=np.uint8)
    board[64, 64] = 255
    patch = crop(board, make_candidate(60, 60, 68, 68), size=16)
    assert patch[8, 8] == 255


def test_build_patch_stacks_template_test_and_difference(blank_board):
    test = blank_board.copy()
    test[60:70, 60:70] = 255

    patch = build_patch(blank_board, test, make_candidate(60, 60, 70, 70))

    assert patch.shape == (3, PATCH_SIZE, PATCH_SIZE)
    assert patch.dtype == np.uint8
    np.testing.assert_array_equal(
        patch[2], np.abs(patch[1].astype(int) - patch[0].astype(int)).astype(np.uint8)
    )


def test_difference_channel_is_empty_for_an_identical_pair(blank_board):
    patch = build_patch(blank_board, blank_board.copy(), make_candidate(60, 60, 70, 70))
    assert patch[2].max() == 0


def test_patch_set_round_trips_through_disk(tmp_path):
    original = PatchSet(
        patches=np.random.randint(0, 255, (5, 3, 8, 8), dtype=np.uint8),
        labels=np.array([0, 1, 2, 0, 1]),
        label_names=["false_call", "open", "short"],
        image_index=np.array([0, 0, 1, 1, 2]),
        boxes=np.zeros((5, 4), dtype=np.int64),
    )
    path = tmp_path / "patches.npz"
    original.save(path)
    loaded = PatchSet.load(path)

    np.testing.assert_array_equal(loaded.patches, original.patches)
    np.testing.assert_array_equal(loaded.labels, original.labels)
    assert loaded.label_names == original.label_names
    assert len(loaded) == 5
