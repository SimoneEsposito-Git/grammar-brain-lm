from typing import List, Optional
from scipy.stats import zscore
import numpy as np

from .embedding_manager import EmbeddingManager
from .context_manager import ContextManager
from .data_sequence import DataSequence
from .tr_file import TRFile

from .file_io import (
    load_dataseqs,
    load_contexts,
    load_features,
    load_responses,
    save_results,
    ensure_directory_exists,
)
from .preprocessing import *
from .validation import validate_embeddings, check_data_alignment
from .config import *


def load_data(
    subjects: List[str],
    modality: str,
    mode: str,
    stories: List[str],
    feature_path: str,
    response_path: str,
    dataseq_path: str,
    contexts_file: str,
    embeddings_file: str,
    overwrite_embeddings: bool = False,
    overwrite_contexts: bool = False,
    verbose: bool = False,
    **kwargs
) -> tuple:
    """
    Main entry point for data loading.

    Returns:
        tuple: (R_trn, R_val, F_trn_processed, F_val_processed)
    """
    stories_train = stories[:-1]
    stories_val = stories[-1:]

    F_trn = load_features("trn", feature_path, stories=stories_train)
    F_val = load_features("val", feature_path, stories=stories_val)

    R_trn = load_responses("trn", response_path, subjects, modality, stories=stories_train)
    R_val = load_responses("val", response_path, subjects, modality, stories=stories_val)
    
    if mode == "english1000":
        return R_trn, R_val, F_trn, F_val, stories_train, stories_val

    dataseqs = load_dataseqs(dataseq_path, stories)

    context_mgr = ContextManager(contexts_file)
    contexts = context_mgr.get_or_create_contexts(
        mode=mode,
        stories=stories,
        dataseqs=dataseqs,
        overwrite=overwrite_contexts,
        **kwargs
    )

    # Handle embeddings
    embedding_mgr = EmbeddingManager(embeddings_file)
    embeddings = embedding_mgr.get_or_create_embeddings(
        mode=mode,
        stories=stories,
        dataseqs=dataseqs,
        contexts=contexts,
        overwrite=overwrite_embeddings,
        verbose=verbose,
    )
    # Validate alignment
    # check_data_alignment(F_trn, {mode: embeddings}, stories)

    # Preprocess
    F_trn_processed = prepare_features(F_trn, embeddings, mode, stories_train)
    F_val_processed = prepare_features(F_val, embeddings, mode, stories_val)

    return R_trn, R_val, F_trn_processed, F_val_processed, stories_train, stories_val


def prepare_data(
    R_trn,
    R_val,
    F_trn,
    F_val,
    stories_trn,
    stories_val,
    use_keys,
    delays=np.arange(0, 9, 2),
):
    """
    Execute a complete encoding model pipeline for fMRI/neural data analysis.

    This function performs feature preparation, ridge regression with cross-validation,
    and validation set evaluation to predict neural responses from stimulus features.

    Args:
        R_trn: Training responses (dict or array-like).
            Neural responses for training stories.
        R_val: Validation responses (dict or array-like).
            Neural responses for validation story.
        F_trn: Training features (dict).
            Feature matrices for training stories, keyed by story identifier.
        F_val: Validation features (dict).
            Feature matrices for validation story, keyed by story identifier.
        stories_trn: List of story identifiers for training set.
        stories_val: List of story identifiers for validation set.
        use_keys: List of feature group keys to use in the model.
            Specifies which feature types to include.
        delays: Array-like, default=np.arange(0, 5).
            Time delays (in TRs) to apply to features for hemodynamic lag modeling.

    Returns:
        tuple: A 4-tuple containing:
            - r (ndarray): Pearson correlation coefficients between predicted and actual
              responses for each voxel/channel.
            - r2 (ndarray): Coefficient of determination (R²) for each voxel/channel.
            - r_sig (ndarray): Correlation coefficients masked by FDR-corrected significance
              (alpha=0.05). Non-significant values set to 0.
            - r_score (float): Sum of significant correlations, used as an overall
              performance metric.

    Notes:
        - Features are standardized and delayed before model fitting.
        - Responses are z-scored and trimmed (first 5 TRs removed).
        - Group Ridge regression with cross-validation is performed across training stories.
        - Statistical significance is assessed using Fisher z-transformation and FDR correction.
        - The function assumes specific data structures and compatible backend operations.
    """

    print("Stacking and delaying features for training...", end="")

    # ═══════════════════════════════════════════════════════════════════
    # Input: {story_id: {feature_key: (n_tr × n_features)}}
    # ═══════════════════════════════════════════════════════════════════
    X_trn = stack_features(F_trn, stories_trn, use_keys, standardize=True)
    # ───────────────────────────────────────────────────────────────────
    # Output: {story_id: (n_tr × total_n_features)}
    # ───────────────────────────────────────────────────────────────────

    X_trn = delay_features(X_trn, stories_trn, delays=delays)
    # ───────────────────────────────────────────────────────────────────
    # Output: {story_id: (n_tr × total_n_features * n_delays)}
    # ───────────────────────────────────────────────────────────────────

    X_trn = stack_stories(X_trn, stories_trn)
    # ───────────────────────────────────────────────────────────────────
    # Output: (total_n_tr × total_n_features * n_delays)
    # ───────────────────────────────────────────────────────────────────
    print(" ✓")

    print("Stacking and delaying features for validation...", end="")
    X_val = stack_features(F_val, stories_val, use_keys, standardize=True)
    X_val = delay_features(X_val, stories_val, delays)
    X_val = stack_stories(X_val, stories_val)
    print(" ✓")

    print("Stacking responses...", end="")
    Y_trn, lens = stack_responses(R_trn, stories_trn, 5, standardize=True)
    # ───────────────────────────────────────────────────────────────────
    # Output: Y_trn = (total_n_tr × n_voxels)
    # ───────────────────────────────────────────────────────────────────

    Y_val = zscore(R_val[stories_val[0]].mean(0)[5:])
    # ───────────────────────────────────────────────────────────────────
    # Output: Y_val = (n_tr × n_voxels)
    # ───────────────────────────────────────────────────────────────────
    print(" ✓")

    print("Building feature groups...", end="")
    groups = build_feature_groups(F_trn[stories_trn[0]], use_keys, n_delays=len(delays))
    story_ids = np.concatenate([np.full(Ld, i, dtype=int) for i, Ld in enumerate(lens)])
    print(" ✓")

    return X_trn, Y_trn, X_val, Y_val, groups, story_ids


# Export main components
__all__ = [
    "load_data",
    "prepare_data",
    "EmbeddingManager",
    "TRFile",
    "DataSequence",
]
