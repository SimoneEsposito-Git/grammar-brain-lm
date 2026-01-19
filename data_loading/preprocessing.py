import numpy as np
from typing import Dict, List


def prepare_features(
    F: np.ndarray, embeddings: Dict, mode: str, stories: List
) -> np.ndarray:
    """Preprocess training data with embeddings."""
    F_ = F.copy()

    for story in stories:
        # Only try to merge if the story exists in the target dictionary
        F_[story][mode] = embeddings[story]
    return F_


def stack_features(F, stories, keys, standardize=True):
    """Stack selected features horizontally with optional standardization.

    Args:
        F: Feature dictionary by story.
        stories: List of story identifiers.
        keys: List of feature keys to stack.
        standardize: Whether to z-score normalize. Defaults to True.

    Returns:
        Dictionary mapping stories to stacked feature arrays.
    """
    blocks = {}
    for s in stories:
        X = np.hstack([np.asarray(F[s][k]) for k in keys])
        if standardize:
            X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        blocks[s] = X
    return blocks


def stack_responses(R, stories, trim, standardize=True):
    """Stack response data across stories with optional standardization.

    Args:
        R: Dictionary of response data by story.
        stories: List of story identifiers to stack.
        trim: Number of initial time points to trim.
        standardize: Whether to z-score normalize. Defaults to True.

    Returns:
        Tuple of (stacked responses array, array of story lengths).
    """
    Ys, lens = {}, {}
    for subject in R.keys():
        Ys[subject] = []
        lens[subject] = []
        for s in stories:
            Y = np.asarray(R[subject][s][trim:])
            if standardize:
                Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)
            Ys[subject].append(np.nan_to_num(Y))
            lens[subject].append(Y.shape[0])
        
    return {subject: np.vstack(Ys[subject]) for subject in Ys}, {subject: np.array(lens[subject]) for subject in lens}


def delay_features(X, stories, delays, circpad=False):
    """Create temporally delayed versions of features.

    Args:
        X: Dictionary of feature arrays by story.
        stories: List of story identifiers.
        delays: List or array of delay values (in time points).
        circpad: Whether to use circular padding. Defaults to False.

    Returns:
        Dictionary mapping stories to delayed feature arrays.
    """
    X_d = {}
    for s in stories:
        stim = X[s]
        nt, ndim = stim.shape
        dstims = []
        for di, d in enumerate(delays):
            dstim = np.zeros((nt, ndim))
            if d < 0:  ## negative delay
                dstim[:d, :] = stim[-d:, :]
                if circpad:
                    dstim[d:, :] = stim[:-d, :]
            elif d > 0:
                dstim[d:, :] = stim[:-d, :]
                if circpad:
                    dstim[:d, :] = stim[-d:, :]
            else:  ## d==0
                dstim = stim.copy()
            dstims.append(dstim)
        X_d[s] = np.hstack(dstims)
    return X_d


def stack_stories(X, stories):
    """Stack feature arrays across multiple stories.

    Args:
        X: Feature array or dictionary.
        stories: List of story identifiers.
        delays: Delay parameters (not used in current implementation).

    Returns:
        Vertically stacked array of all stories.
    """
    blocks = []
    for s in stories:
        blocks.append(X[s])
    return np.vstack(blocks)


def build_feature_groups(F_one_story, keys, n_delays):
    """Build group indices for delayed feature columns.

    Args:
        F_one_story: Feature dictionary for a single story.
        keys: List of feature keys.
        n_delays: Number of delay taps.

    Returns:
        Array of group indices for each column in delayed design matrix.
    """
    group_idx_raw = []
    for gi, k in enumerate(keys):
        group_idx_raw.append(np.full(F_one_story[k].shape[1], gi, dtype=int))
    group_idx_raw = np.hstack(group_idx_raw)
    groups_delayed = np.repeat(group_idx_raw, n_delays)
    return groups_delayed


def build_kernels_from_groups(X, groups_delayed):
    """Build separate kernel matrices for each feature group.

    Args:
        X: Feature design matrix.
        groups_delayed: Array of group indices for each column.

    Returns:
        3D array of kernel matrices stacked along first axis.
    """
    kernels = []
    unique_groups = np.unique(groups_delayed)

    for group in unique_groups:
        mask = groups_delayed == group
        X_group = X[:, mask]
        kernel = X_group @ X_group.T
        kernels.append(kernel)

    return np.stack(kernels, axis=0)


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize feature vectors."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    return (features - mean) / (std + 1e-8)
