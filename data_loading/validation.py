import numpy as np
from typing import List


def validate_stories(stories: List[str]) -> bool:
    """Check if story list is valid."""
    if not stories:
        raise ValueError("Story list is empty")
    if len(stories) != len(set(stories)):
        raise ValueError("Duplicate stories found")
    return True


def validate_embeddings(embeddings: dict, expected_dim: int = None) -> bool:
    """Validate embedding structure and dimensions."""
    if not embeddings:
        return False

    for mode, mode_emb in embeddings.items():
        if not isinstance(mode_emb, dict):
            raise ValueError(f"Embeddings for mode {mode} must be a dict")

        if expected_dim:
            for story, emb in mode_emb.items():
                if len(emb) != expected_dim:
                    raise ValueError(f"Story {story} has wrong embedding dimension")
    return True


def check_data_alignment(features: np.ndarray, embeddings: dict, stories: List[str]):
    """Ensure features, embeddings, and stories are properly aligned."""
    if len(features) != len(stories):
        raise ValueError(
            f"Feature count ({len(features)}) doesn't match story count ({len(stories)})"
        )


def validate_prepared_data(
    X_trn: np.ndarray,
    Y_trn: np.ndarray,
    groups: np.ndarray,
    story_ids: np.ndarray,
    use_keys: List[str],
    verbose: bool = False,
):
    """
    Validate prepared training data for pipeline execution.

    Args:
        X_trn: Training features array (n_samples, n_features)
        Y_trn: Training responses array (n_samples, n_targets)
        groups: Feature group indices (n_features,)
        story_ids: Story identifiers for each sample (n_samples,)
        use_keys: List of feature group keys
        verbose: Whether to print validation details

    Raises:
        AssertionError: If validation fails
    """
    if verbose:
        print("\nValidating prepared data...")

    n_samples_X, n_features = X_trn.shape
    n_samples_Y, n_targets = Y_trn.shape

    # Validate shapes and data integrity
    assert (
        n_samples_X == n_samples_Y
    ), f"Sample count mismatch: X_trn has {n_samples_X} samples, Y_trn has {n_samples_Y}"
    assert (
        len(story_ids) == n_samples_Y
    ), f"story_ids length {len(story_ids)} doesn't match Y_trn samples {n_samples_Y}"
    assert (
        len(groups) == n_features
    ), f"groups length {len(groups)} doesn't match X_trn features {n_features}"

    # Validate groups
    unique_groups = np.unique(groups)
    if verbose:
        print(f"  Unique groups: {unique_groups}")
        print(
            f"  Expected groups (0 to {len(use_keys)-1}): {list(range(len(use_keys)))}"
        )

    assert len(unique_groups) == len(
        use_keys
    ), f"Number of unique groups {len(unique_groups)} doesn't match use_keys {len(use_keys)}"
    assert np.all(
        unique_groups == np.arange(len(use_keys))
    ), f"Group indices {unique_groups} don't match expected range [0, {len(use_keys)-1}]"

    # Validate story_ids
    unique_stories = np.unique(story_ids)
    if verbose:
        print(f"  Unique story IDs: {unique_stories}")
        print(f"  Number of unique stories: {len(unique_stories)}")

    assert (
        len(unique_stories) >= 2
    ), f"Need at least 2 unique stories for CV, found {len(unique_stories)}"

    # Check for NaN/Inf values
    if np.any(np.isnan(X_trn)) or np.any(np.isinf(X_trn)):
        n_nan = np.sum(np.isnan(X_trn))
        n_inf = np.sum(np.isinf(X_trn))
        if verbose:
            print("  WARNING: X_trn contains NaN or Inf values!")
            print(f"    NaN count: {n_nan}, Inf count: {n_inf}")

    if np.any(np.isnan(Y_trn)) or np.any(np.isinf(Y_trn)):
        n_nan = np.sum(np.isnan(Y_trn))
        n_inf = np.sum(np.isinf(Y_trn))
        if verbose:
            print("  WARNING: Y_trn contains NaN or Inf values!")
            print(f"    NaN count: {n_nan}, Inf count: {n_inf}")

    # Check feature distribution per group
    if verbose:
        print("\nFeatures per group:")
    for gi, key in enumerate(use_keys):
        n_feats = np.sum(groups == gi)
        if verbose:
            print(f"  Group {gi} ({key}): {n_feats} features")
        assert n_feats > 0, f"Group {gi} ({key}) has no features!"

    if verbose:
        print("Validation complete ✓\n")
