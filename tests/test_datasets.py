import numpy as np
import pytest

from src.datasets import is_indoor, list_rirs, rir_label
from src.io import save_wav


def test_rir_label_extracts_room_type_from_filename():
    assert rir_label("h001_Bedroom_65txts.wav") == "Bedroom"


def test_rir_label_ignores_path_and_sub_description():
    name = "/data/public/MIT_Survey/h003_Office_LargeBrickWalledOpenPlanOffice_56txts.wav"
    assert rir_label(name) == "Office"


def test_is_indoor_true_for_enclosed_rooms():
    assert is_indoor("Bedroom")
    assert is_indoor("Classroom")
    assert is_indoor("Office")


def test_is_indoor_false_for_outdoor_and_transit():
    assert not is_indoor("Outside")
    assert not is_indoor("MITCampus")
    assert not is_indoor("StreetsOfBoston")
    assert not is_indoor("ParkingLot")
    assert not is_indoor("Car")


def test_list_rirs_returns_only_indoor_sorted(tmp_path):
    for name in ["h002_Bedroom_x.wav", "h001_Office_x.wav", "h003_Outside_x.wav"]:
        save_wav(str(tmp_path / name), np.zeros(64), 16000)

    paths = list_rirs(str(tmp_path))

    assert [p.split("\\")[-1].split("/")[-1] for p in paths] == [
        "h001_Office_x.wav",
        "h002_Bedroom_x.wav",
    ]


def test_list_rirs_includes_outdoor_when_indoor_only_false(tmp_path):
    for name in ["h001_Office_x.wav", "h003_Outside_x.wav"]:
        save_wav(str(tmp_path / name), np.zeros(64), 16000)

    paths = list_rirs(str(tmp_path), indoor_only=False)

    assert len(paths) == 2


def test_list_rirs_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_rirs(str(tmp_path / "does_not_exist"))
