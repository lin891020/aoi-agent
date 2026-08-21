import pytest

from aoi_agent.data.deeppcb import CLASS_NAMES, Annotation, load_split


def test_class_ids_match_the_dataset_specification():
    assert CLASS_NAMES == {
        1: "open", 2: "short", 3: "mousebite",
        4: "spur", 5: "copper", 6: "pin-hole",
    }


def test_annotation_exposes_its_box_and_class():
    annotation = Annotation(10, 20, 30, 40, 3)
    assert annotation.box == (10, 20, 30, 40)
    assert annotation.class_name == "mousebite"


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError, match="trainval"):
        load_split("train")


@pytest.mark.dataset
def test_official_splits_have_the_documented_sizes():
    assert len(load_split("trainval")) == 1000
    assert len(load_split("test")) == 500


@pytest.mark.dataset
def test_every_pair_resolves_to_files_on_disk():
    for pair in load_split("test")[:20]:
        assert pair.template_path.exists(), pair.template_path
        assert pair.test_path.exists(), pair.test_path
        assert pair.annotation_path.exists(), pair.annotation_path


@pytest.mark.dataset
def test_annotations_parse_into_valid_boxes():
    for pair in load_split("test")[:20]:
        annotations = pair.load_annotations()
        assert annotations, f"{pair.stem} has no annotations"
        for annotation in annotations:
            assert annotation.x1 < annotation.x2
            assert annotation.y1 < annotation.y2
            assert annotation.class_id in CLASS_NAMES


@pytest.mark.dataset
def test_template_and_test_images_are_the_same_shape():
    pair = load_split("test")[0]
    assert pair.load_template().shape == pair.load_test().shape == (640, 640)
