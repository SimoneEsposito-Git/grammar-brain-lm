import numpy as np
import pytest

from data_loading.preprocessing import (
    build_feature_groups,
    build_kernels_from_groups,
    delay_features,
    normalize_features,
    prepare_features,
    stack_features,
    stack_responses,
    stack_stories,
)


def test_normalize_features_zero_mean_unit_std():
    features = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    normalized = normalize_features(features)
    assert np.allclose(normalized.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(normalized.std(axis=0), 1.0, atol=1e-8)


def test_stack_features_concatenates_keys_in_order():
    F = {"s1": {"a": np.array([[1.0], [2.0]]), "b": np.array([[3.0], [4.0]])}}
    stacked = stack_features(F, stories=["s1"], keys=["a", "b"], standardize=False)
    assert np.array_equal(stacked["s1"], np.array([[1.0, 3.0], [2.0, 4.0]]))


def test_stack_features_standardizes_when_requested():
    F = {"s1": {"a": np.array([[1.0], [3.0]])}}
    stacked = stack_features(F, stories=["s1"], keys=["a"], standardize=True)
    assert np.allclose(stacked["s1"].mean(axis=0), 0.0, atol=1e-8)


def test_stack_stories_vstacks_in_given_order():
    X = {"s1": np.array([[1.0]]), "s2": np.array([[2.0]])}
    result = stack_stories(X, stories=["s2", "s1"])
    assert np.array_equal(result, np.array([[2.0], [1.0]]))


def test_delay_features_zero_delay_is_unchanged():
    X = {"s1": np.array([[1.0, 2.0], [3.0, 4.0]])}
    result = delay_features(X, stories=["s1"], delays=[0])
    assert np.array_equal(result["s1"], X["s1"])


def test_delay_features_positive_delay_shifts_forward():
    X = {"s1": np.array([[1.0], [2.0], [3.0]])}
    result = delay_features(X, stories=["s1"], delays=[1])
    # Row 0 has no history so stays 0; later rows hold the previous timestep's value.
    assert np.array_equal(result["s1"], np.array([[0.0], [1.0], [2.0]]))


def test_stack_responses_trims_and_reports_lengths():
    R = {"subj1": {"s1": np.arange(10.0).reshape(10, 1)}}
    stacked, lens = stack_responses(R, stories=["s1"], trim=2, standardize=False)
    assert stacked["subj1"].shape[0] == 8
    assert lens["subj1"][0] == 8


def test_build_feature_groups_repeats_indices_per_delay():
    F_one_story = {"a": np.zeros((5, 2)), "b": np.zeros((5, 3))}
    groups = build_feature_groups(F_one_story, keys=["a", "b"], n_delays=2)
    # 2 columns of group 0 + 3 columns of group 1, each repeated for 2 delays.
    assert list(groups) == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]


def test_build_kernels_from_groups_one_kernel_per_group():
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    groups = np.array([0, 0, 1])
    kernels = build_kernels_from_groups(X, groups)
    assert kernels.shape == (2, 2, 2)
    expected_group0 = X[:, :2] @ X[:, :2].T
    assert np.allclose(kernels[0], expected_group0)


def test_prepare_features_merges_embeddings_into_mode_key():
    F = {"s1": {"other": np.zeros(3)}}
    embeddings = {"s1": np.array([1.0, 2.0, 3.0])}
    result = prepare_features(F, embeddings, mode="emb", stories=["s1"])
    assert np.array_equal(result["s1"]["emb"], embeddings["s1"])
    # F.copy() is shallow, so per-story dicts are shared with the input; this
    # also mutates F["s1"] in place, which is worth knowing about at call sites.
    assert "emb" in F["s1"]
