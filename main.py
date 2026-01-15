import os
import numpy as np
import argparse

from himalaya.kernel_ridge import MultipleKernelRidgeCV, linear_kernel
from himalaya.kernel_ridge import (
    solve_multiple_kernel_ridge_random_search,
    predict_and_score_weighted_kernel_ridge,
)
from himalaya.scoring import r2_score_split, correlation_score_split
from himalaya.backend import set_backend
from sklearn.model_selection import LeaveOneGroupOut
from typing import Dict, List, Union
from scipy.stats import zscore, pearsonr, norm
from statsmodels.stats.multitest import fdrcorrection

import scripts.plotting_utils as pu

from data_loading import load_data, prepare_data, config
from data_loading.validation import validate_prepared_data

try:
    backend = set_backend("torch_cuda")
except Exception as e:
    print(f"Warning: Failed to set CUDA backend: {e}. Falling back to numpy backend.")
    backend = set_backend("numpy")


def pipeline(
    R_trn,
    R_val,
    F_trn,
    F_val,
    stories_trn,
    stories_val,
    use_keys,
    delays=np.arange(0, 5),
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

    X_trn, Y_trn, X_val, Y_val, groups, story_ids = prepare_data(
        R_trn,
        R_val,
        F_trn,
        F_val,
        stories_trn,
        stories_val,
        use_keys,
        delays=delays,
    )

    # Delegate validation to validation module
    # validate_prepared_data(X_trn, Y_trn, groups, story_ids, use_keys)

    print("Performing Group Ridge CV...", end="")
    results = perform_group_ridge_cv(
        X_trn,
        Y_trn,
        groups,
        story_ids,
        alphas=np.logspace(5, 15, 11),
        use_keys=use_keys,
    )
    print(" ✓")

    print("Processing results...", end="")
    deltas = backend.to_numpy(results[0])
    dual_weights = backend.to_numpy(results[1])
    cv_scores = backend.to_numpy(results[2])
    print(" ✓")

    print("Computing kernels for validation...", end="")
    Ks_val = []
    for gi, key in enumerate(use_keys):
        cols = np.where(groups == gi)[0]
        K_val = X_val[:, cols] @ X_trn[:, cols].T
        Ks_val.append(K_val.astype(np.float32))
    print(" ✓")

    Ks_val = np.stack(Ks_val)
    Y_val = backend.asarray(Y_val, dtype=backend.float32)
    Ks_val = backend.asarray(Ks_val, dtype=backend.float32)

    print("Calculating correlations...", end="")
    r, r2 = get_correlation(Ks_val, dual_weights, deltas, Y_val)
    print(" ✓")

    T = X_val.shape[0]
    z = r[0] * np.sqrt(T - 3)
    pvals = 1 - norm.cdf(z)

    print("Applying FDR correction...", end="")
    sig_mask, _ = fdrcorrection(pvals, alpha=0.05)
    print(" ✓")

    r_sig = r[0] * sig_mask
    r_score = np.nansum(r_sig)
    print("Pipeline execution complete.")
    return r[0], r2[0], r_sig, r_score


def get_correlation(Ks_val, dual_weights, deltas, Y_val):
    r2 = backend.to_numpy(
        predict_and_score_weighted_kernel_ridge(
            Ks_val,
            dual_weights,
            deltas,
            Y_val,
            split=True,
            n_targets_batch=512,
            score_func=r2_score_split,
        )
    )

    r = backend.to_numpy(
        predict_and_score_weighted_kernel_ridge(
            Ks_val,
            dual_weights,
            deltas,
            Y_val,
            split=True,
            n_targets_batch=512,
            score_func=correlation_score_split,
        )
    )

    return r, r2


def perform_group_ridge_cv(
    X_train, Y_train, groups_delayed, story_ids, alphas, use_keys
):
    """
    Perform group ridge regression with cross-validation using multiple kernel ridge regression.
    This function splits the input data into groups based on delayed features, converts them into
    kernel matrices, and performs a random search for optimal alpha parameters using multiple
    kernel ridge regression with cross-validation.
    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
        Training input data matrix.
    Y_train : array-like, shape (n_samples, n_targets)
        Training target values.
    groups_delayed : array-like, shape (n_features,)
        Group indices for each feature, indicating which group each feature belongs to.
    story_ids : array-like, shape (n_samples,)
        Group labels for cross-validation splits using Leave-One-Group-Out strategy.
    alphas : array-like
        Regularization parameter values to search over.
    use_keys : list
        List of keys corresponding to each group to be used in the analysis.
    Returns
    -------
    results : dict
        Dictionary containing the results from the multiple kernel ridge regression,
        including optimal alpha values and dual weights. The exact structure depends
        on the output of solve_multiple_kernel_ridge_random_search.
    Notes
    -----
    - The function converts input data to float32 for computational efficiency.
    - Cross-validation is performed using Leave-One-Group-Out strategy based on story_ids.
    - Kernel matrices are computed as the dot product of each group's features with itself.
    - The random search uses batching for both targets and alphas to manage memory usage.
    """

    Xs_train = []
    for gi, key in enumerate(use_keys):
        cols = np.where(groups_delayed == gi)[0]
        Xs_train.append(X_train[:, cols])

    print("Fitting MultipleKernelRidgeCV model...")

    Xs_train = [backend.asarray(X_train, dtype="float32") for X_train in Xs_train]
    Ks_train = backend.stack([X_train @ X_train.T for X_train in Xs_train])
    Ks_train = backend.asarray(Ks_train, dtype=backend.float32)
    Y_train = backend.asarray(Y_train, dtype=backend.float32)

    # cv_splits = list(
    #     LeaveOneGroupOut().split(np.zeros(Ks_train.shape[1]), groups=story_ids)
    # )
    # 
    # print(f"Created {len(cv_splits)} CV splits.")

    results = solve_multiple_kernel_ridge_random_search(
        Ks=Ks_train,
        Y=Y_train,
        alphas=alphas,
        n_targets_batch=512,
        return_weights="dual",
        n_alphas_batch=20,
        n_targets_batch_refit=200,
        jitter_alphas=True,
    )

    print("Model fitting complete.")
    return results


def main(
    subject,
    modality,
    mode,
    stories,
    nuis_listening,
    nuis_reading,
    fdir,
    trfile_dir,
    transcript_dir,
    verbose=False,
    **kwargs,
):
    """
    Execute the main analysis pipeline for a given subject and modality.
    This function loads neural response data and stimulus features, prepares them according
    to the specified mode, runs a correlation analysis pipeline, and saves/visualizes the results.
    Args:
        subject (str): Subject identifier for loading neural response data.
        modality (str): Data modality, either "listening" or "reading".
        mode (str): Feature extraction mode to use for analysis.
        nuis_listening (list): List of nuisance regressor keys to use for listening modality.
        nuis_reading (list): List of nuisance regressor keys to use for reading modality.
        fdir (str): Base directory path for loading/saving files.
        trfile_dir (str): Directory path containing TR (repetition time) files.
        transcript_dir (str): Directory path containing transcript files.
    Returns:
        None: Results are saved to disk as .npz files and correlation plots.
    Side Effects:
        - Prints progress messages to stdout
        - Creates 'outputs' directory if it doesn't exist
        - Saves results to 'outputs/results_{subject}_{modality}_{mode}.npz'
        - Generates and saves correlation flatmap visualizations
    Notes:
        - Uses global variables STORIES and STORY_NAMES for story information
        - Training data uses first 10 stories, validation uses 11th story
        - The score is calculated as the sum of significant correlations
    """

    R_trn, R_val, F_trn, F_val, stories_trn, stories_val = load_data(
        [subject],
        modality,
        mode,
        stories,
        config.DEFAULT_FEATURE_PATH,
        config.DEFAULT_RESPONSE_PATH,
        config.DEFAULT_DATASEQ_PATH,
        config.DEFAULT_CONTEXTS_FILE,
        config.DEFAULT_EMBEDDINGS_FILE,
        verbose=verbose,
        **kwargs,
    )

    use_keys = [mode] + (nuis_listening if modality == "listening" else nuis_reading)

    r, r2, r_sig, score = pipeline(
        R_trn[subject],
        R_val[subject],
        F_trn,
        F_val,
        stories_trn,
        stories_val,
        use_keys,
    )

    print(
        f"Total significant correlation score for subject {subject}, modality {modality}, mode {mode}: {score}"
    )

    # Save results to npz file
    results_file = (
        f"{config.DEFAULT_OUTPUT_PATH}/results/{subject}_{modality}_{mode}.npz"
    )
    print(f"Saving results to {results_file}")
    try:
        np.savez(results_file, score=score, r=r, r_sig=r_sig, r2=r2)
    except FileNotFoundError as e:
        print(f"Error saving results: {e}. Please ensure the directory exists.")
        return
    pu.plot_correlation_on_flatmap(
        subject, modality, mode, r, r_sig, config.DEFAULT_MAPPER_PATH
    )

    print("Script execution finished.")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Run encoding model pipeline for fMRI analysis"
        )
        parser.add_argument(
            "--subject",
            type=str,
            default="subject07",
            help="Subject identifier (default: subject07)",
        )
        parser.add_argument(
            "--modality",
            type=str,
            default="listening",
            choices=["listening", "reading"],
            help="Data modality (default: listening)",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="baseline",
            help="Feature extraction mode (default: baseline)",
        )
        parser.add_argument(
            "--trfile-dir",
            type=str,
            default=os.path.join("stimuli", "textgrids", "trfiles"),
            help="Directory containing TR files",
        )
        parser.add_argument(
            "--transcript-dir",
            type=str,
            default=os.path.join("stimuli", "textgrids", "transcripts"),
            help="Directory containing transcript files",
        )
        parser.add_argument(
            "--fdir", type=str, default="./", help="Base directory path (default: ./)"
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose output (default: False)",
        )
        parser.add_argument(
            "--kwargs",
            type=str,
            default="{}",
            help="Additional keyword arguments as a JSON string (default: '{}')",
        )

        args = parser.parse_args()

        main(
            args.subject,
            args.modality,
            args.mode,
            config.STORIES,
            config.NUIS_LISTENING,
            config.NUIS_READING,
            args.fdir,
            args.trfile_dir,
            args.transcript_dir,
            args.verbose,
            **eval(args.kwargs),
        )
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
