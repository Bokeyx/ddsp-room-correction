import numpy as np
import pytest
import scipy.io as sio

from src.datasets import is_indoor, list_air_rirs, list_rirs, load_air_mat, rir_label
from src.io import save_wav


def _write_air_mat(path, h=(0.0,), fs=48000, room="office", channel=0):
    """Write a minimal Aachen-AIR-style .mat (air_info struct + h_air vector)."""
    sio.savemat(str(path), {
        "air_info": {"fs": fs, "room": room, "channel": channel},
        "h_air": np.asarray(h, dtype=np.float64).reshape(1, -1),
    })


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


def test_load_air_mat_returns_mono_ir_with_fs_and_room(tmp_path):
    f = tmp_path / "air_binaural_office_0_0_1.mat"
    _write_air_mat(f, np.linspace(1.0, 0.0, 100), fs=48000, room="office")

    h, fs, room = load_air_mat(str(f))

    assert h.ndim == 1 and len(h) == 100
    assert fs == 48000
    assert room == "office"


def test_list_air_rirs_keeps_listening_rooms_drops_stairway_and_phone(tmp_path):
    for name in [
        "air_binaural_office_0_0_1.mat",
        "air_binaural_lecture_0_0_1.mat",
        "air_binaural_stairway_0_0_1.mat",   # excluded: not a listening room
        "air_phone_BT_office_hhp_x.mat",     # excluded: band-limited phone
    ]:
        _write_air_mat(tmp_path / name)

    names = [p.replace("\\", "/").split("/")[-1] for p in list_air_rirs(str(tmp_path))]

    assert names == ["air_binaural_lecture_0_0_1.mat", "air_binaural_office_0_0_1.mat"]


def test_list_air_rirs_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_air_rirs(str(tmp_path / "does_not_exist"))
