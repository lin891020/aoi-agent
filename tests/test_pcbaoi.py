"""PCB-AoI as the detector sees it: stems, splits, and the label round-trip.

The one thing a dataset adapter for an augmented set can get quietly wrong is
the split. The augmentation folder is six transforms of each original under
the original's stem plus a suffix; a split drawn over *images* puts a rotated
copy of a validation board into training and the validation score rises for
no reason. So the split is over base stems, and this file holds that the
suffix grammar is understood, that no base stem sits on both sides, and that
a YOLO label survives a round trip through the normalisation.

The pure tests need no download. The rest are behind ``-m dataset`` and skip
when `data/PCB-AoI` is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aoi_agent.data import deeppcb, pcbaoi
from aoi_agent.data.pcbaoi import Box, Item, base_stem, from_yolo_line, to_yolo_line

# ---- pure -----------------------------------------------------------------


@pytest.mark.parametrize("stem, base", [
    ("20161019-SPI-AOI-1", "20161019-SPI-AOI-1"),
    ("20161019-SPI-AOI-1_180", "20161019-SPI-AOI-1"),
    ("20161019-SPI-AOI-1_shuiping", "20161019-SPI-AOI-1"),
    ("20170314-SPI-AOI-19_suofang", "20170314-SPI-AOI-19"),
])
def test_a_transforms_stem_names_its_original(stem, base):
    assert base_stem(stem) == base


def test_a_stem_that_is_not_pcbaoi_is_refused():
    with pytest.raises(ValueError, match="not a PCB-AoI stem"):
        base_stem("20085294")


def test_the_classes_are_the_datasets_own_and_not_deeppcbs():
    assert set(pcbaoi.CLASS_NAMES) == {"Bad_podu", "Bad_qiaojiao"}
    assert not set(pcbaoi.CLASS_NAMES) & set(deeppcb.CLASS_NAMES.values())


def test_a_yolo_label_survives_the_round_trip():
    box = Box(30, 51, 46, 79, "Bad_podu")
    line = to_yolo_line(box, 600, 600)
    assert line.startswith("0 ")
    back = from_yolo_line(line, 600, 600)
    assert (back.x1, back.y1, back.x2, back.y2) == (30, 51, 46, 79)
    assert back.class_name == "Bad_podu"


def _fake_items(tmp_path, stems):
    items = []
    for stem in stems:
        items.append(Item(stem=stem, image_path=tmp_path / f"{stem}.jpeg",
                          annotation_path=tmp_path / f"{stem}.xml"))
    return items


def test_the_split_keeps_every_transform_with_its_original(tmp_path):
    stems = [f"2016101{d}-SPI-AOI-{n}" for d in range(10) for n in range(3)]
    items = _fake_items(tmp_path, stems + [s + "_90" for s in stems] + [s + "_shuzhi" for s in stems])
    train, val = pcbaoi.split_by_stem(items, validation_share=0.25)

    assert pcbaoi.leaks(train, val) == set()
    assert len(train) + len(val) == len(items)
    assert 0 < len(val) < len(items)
    # every image of a held-out base is on the validation side, all six of them
    for base in {i.base_stem for i in val}:
        assert sum(1 for i in val if i.base_stem == base) == 3


def test_the_split_is_the_same_split_on_a_second_run(tmp_path):
    items = _fake_items(tmp_path, [f"2016101{d}-SPI-AOI-{n}" for d in range(10) for n in range(4)])
    a = pcbaoi.split_by_stem(items)
    b = pcbaoi.split_by_stem(items)
    assert [i.stem for i in a[1]] == [i.stem for i in b[1]]


# ---- against the files -----------------------------------------------------


@pytest.fixture(scope="module")
def shipped():
    if not (pcbaoi.DEFAULT_ROOT / "test_data" / "index.txt").exists():
        pytest.skip("PCB-AoI not downloaded; see data/pcbaoi.py")
    return {s: pcbaoi.load(s) for s in ("train_data", "train_data_augmentation", "test_data")}


@pytest.mark.dataset
def test_the_shipped_counts_are_what_the_inventory_found(shipped):
    assert len(shipped["train_data"]) == 173
    assert len(shipped["train_data_augmentation"]) == 1211
    assert len(shipped["test_data"]) == 60


@pytest.mark.dataset
def test_no_augmentation_of_a_test_board_exists(shipped):
    """The property the whole benchmark rests on."""
    assert pcbaoi.leaks(shipped["train_data_augmentation"], shipped["test_data"]) == set()
    assert pcbaoi.leaks(shipped["train_data"], shipped["test_data"]) == set()


@pytest.mark.dataset
def test_every_box_is_one_of_the_two_classes_and_inside_the_frame(shipped):
    for item in shipped["test_data"]:
        width, height = item.image_size()
        for b in item.load_boxes():
            assert b.class_name in pcbaoi.CLASS_NAMES
            assert 0 <= b.x1 < b.x2 <= width and 0 <= b.y1 < b.y2 <= height


@pytest.mark.dataset
def test_export_writes_the_yolo_layout_and_touches_nothing_shipped(shipped, tmp_path):
    train, val = pcbaoi.split_by_stem(shipped["train_data_augmentation"])
    before = sorted(p.name for p in (pcbaoi.DEFAULT_ROOT / "test_data" / "JPEGImages").iterdir())
    yaml = pcbaoi.export(train[:5], val[:3], shipped["test_data"][:2], out=tmp_path / "yolo")

    assert yaml.exists() and "names:" in yaml.read_text()
    assert len(list((tmp_path / "yolo" / "images" / "train").iterdir())) == 5
    assert len(list((tmp_path / "yolo" / "labels" / "test").iterdir())) == 2
    label = next((tmp_path / "yolo" / "labels" / "test").iterdir()).read_text().splitlines()
    assert all(len(l.split()) == 5 for l in label)
    after = sorted(p.name for p in (pcbaoi.DEFAULT_ROOT / "test_data" / "JPEGImages").iterdir())
    assert before == after
