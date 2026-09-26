import numpy as np

from data_loading.data_sequence import DataSequence


def make_sequence():
    # 6 data points split into 3 chunks of 2.
    data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    return DataSequence(data, split_inds=[2, 4])


def test_chunks_splits_data_at_split_inds():
    ds = make_sequence()
    chunks = ds.chunks()
    assert len(chunks) == 3
    assert np.array_equal(chunks[0], [[1.0], [2.0]])
    assert np.array_equal(chunks[1], [[3.0], [4.0]])
    assert np.array_equal(chunks[2], [[5.0], [6.0]])


def test_data_to_chunk_ind_finds_containing_chunk():
    ds = make_sequence()
    assert ds.data_to_chunk_ind(0) == 0
    assert ds.data_to_chunk_ind(3) == 1
    assert ds.data_to_chunk_ind(5) == 2


def test_chunk_to_data_ind_returns_indices_for_chunk():
    ds = make_sequence()
    assert ds.chunk_to_data_ind(1) == [2, 3]


def test_chunksums_default_sums_each_chunk():
    ds = make_sequence()
    sums = ds.chunksums()
    assert np.array_equal(sums, [[3.0], [7.0], [11.0]])


def test_chunksums_mean_averages_each_chunk():
    ds = make_sequence()
    means = ds.chunksums(interp="mean")
    assert np.array_equal(means, [[1.5], [3.5], [5.5]])


def test_copy_produces_independent_split_inds():
    ds = make_sequence()
    ds_copy = ds.copy()
    ds_copy.split_inds.append(99)
    assert ds.split_inds == [2, 4]


def test_from_chunks_round_trips_with_chunks():
    original_chunks = [[1, 2], [3, 4, 5], [6]]
    ds = DataSequence.from_chunks(original_chunks)
    assert ds.data == [1, 2, 3, 4, 5, 6]
    assert list(ds.split_inds) == [2, 5]
