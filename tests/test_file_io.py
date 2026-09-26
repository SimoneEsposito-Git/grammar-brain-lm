import pickle

import numpy as np

from data_loading.file_io import (
    ensure_directory_exists,
    load_dataseqs,
    load_results,
    save_results,
)


def test_ensure_directory_exists_creates_parents(tmp_path):
    target = tmp_path / "a" / "b" / "c.txt"
    ensure_directory_exists(str(target))
    assert target.parent.is_dir()


def test_load_results_missing_file_returns_empty_dict(tmp_path):
    assert load_results(tmp_path, "listening", "subject01") == {}


def test_save_then_load_results_roundtrip(tmp_path):
    data = {"r": np.array([1.0, 2.0, 3.0])}
    save_results(tmp_path, "listening", "subject01", data)
    loaded = load_results(tmp_path, "listening", "subject01")
    assert np.array_equal(loaded["r"], data["r"])


def test_load_dataseqs_reads_pickled_objects(tmp_path):
    (tmp_path / "story1.pkl").write_bytes(pickle.dumps({"foo": "bar"}))
    result = load_dataseqs(str(tmp_path), ["story1"])
    assert result["story1"] == {"foo": "bar"}
