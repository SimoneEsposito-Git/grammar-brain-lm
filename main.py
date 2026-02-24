import os
import numpy as np
import argparse
import torch
from tqdm import tqdm 
import time

from himalaya.kernel_ridge import MultipleKernelRidgeCV, linear_kernel
from himalaya.kernel_ridge import (
    solve_multiple_kernel_ridge_random_search,
    predict_and_score_weighted_kernel_ridge,
    predict_weighted_kernel_ridge
)
from himalaya.scoring import r2_score_split, correlation_score_split
from himalaya.backend import set_backend
from sklearn.model_selection import LeaveOneGroupOut
from typing import Dict, List, Union
from scipy.stats import zscore, pearsonr, norm
from statsmodels.stats.multitest import fdrcorrection

import scripts.plotting_utils as pu

from data_loading import load_data, prepare_data, config, load_results, save_results
from data_loading.validation import validate_prepared_data

try:
    backend = set_backend("torch_cuda")
except Exception as e:
    print(f"Warning: Failed to set CUDA backend: {e}. Falling back to numpy backend.")
    backend = set_backend("numpy")
    
def get_scores_and_prediction(results_grr, X_val, X_trn, Y_val, groups, use_keys):
    print("Processing results...", end="")
    deltas = backend.to_numpy(results_grr[0])
    dual_weights = backend.to_numpy(results_grr[1])
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

    predictions = predict_weighted_kernel_ridge(
        Ks_val, dual_weights, deltas, split=False
    )
    #predictions = backend.to_numpy(predictions)
    Y_val = backend.to_numpy(Y_val)
    
    return r[0], r2[0], predictions

def apply_fdr_correction(pvalues):
    print("Applying FDR correction...", end="")
    fdr = {}
    for i in pvalues:
        pvalue = backend.to_numpy(pvalues[i][0])
        rejected, corrected_p_values = fdrcorrection(
            pvalue, alpha=0.05, method="indep", is_sorted=False
        )
        fdr[i] = {
            "corrected_p_values": corrected_p_values,
            "included_voxels_indices": get_bh_included_voxels(
                pvalue, 0.05
            )[0],
            "excluded_voxels_indices": get_bh_excluded_voxels(
                pvalue, 0.05
            )[0],
        }
        
    # sig_mask, _ = fdrcorrection(pvals, alpha=0.05)
    print(" ✓")
    return fdr
    # r_sig = r[0] * 
    # r_score = np.nansum(r_sig)
    # print("Pipeline execution complete.")
    # return r[0], r2[0], r_sig, r_score

def get_bh_excluded_voxels(pvalues: np.ndarray,
        alpha: float):
    num_values = len(pvalues)
    pvalues_sorted = np.sort(pvalues)
    max_p = pvalues_sorted[np.argmax(np.where(pvalues_sorted <= ((np.arange(1, num_values + 1) / num_values) * alpha)))]
    voxels_excluded = np.where(pvalues > max_p)
    return voxels_excluded

def get_bh_included_voxels(pvalues: np.ndarray,
        alpha: float):
    num_values = len(pvalues)
    pvalues_sorted = np.sort(pvalues)
    max_p = pvalues_sorted[np.argmax(np.where(pvalues_sorted <= ((np.arange(1, num_values + 1) / num_values) * alpha)))]
    voxels_included = np.where(pvalues <= max_p)
    return voxels_included

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

def permutation_test(responses_test, predictions, score_func, num_permutations=1000, permutation_block_size=10):
    true_scores = score_func(responses_test, predictions)
    if torch.cuda.is_available():
        predictions = torch.clone(predictions)
        num_get_true_score = torch.zeros(true_scores.shape, device=predictions.device)
    else:
        predictions = np.copy(predictions)
        num_get_true_score = np.zeros(true_scores.shape)
    num_TRs = predictions.shape[0]
    blocks = np.array_split(np.arange(num_TRs), int(num_TRs / permutation_block_size))
    for permutation_num in tqdm(range(num_permutations)):
        _ = np.random.shuffle(blocks)
        permutation_order = np.concatenate(blocks)
        predictions = predictions[permutation_order]
        shuffled_scores = score_func(responses_test, predictions)
        num_get_true_score[shuffled_scores >= true_scores] += 1
    pvalues = num_get_true_score / num_permutations
    return pvalues, true_scores

def perform_group_ridge(X_train, Y_train, groups_delayed, story_ids, alphas, use_keys):
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

def pipeline(
    subjects,
    modality,
    mode,
    stories,
    nuis_listening,
    nuis_reading,
    root_dir,
    output_dir = None,
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
        root_dir (str): Root directory path for data and outputs.
        output_dir (str): Output directory path for saving results.
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
    # verify that kwargs contains the expected keys for overwrite flags

    if "overwrite_contexts" in kwargs and "overwrite_embeddings" in kwargs:
        overwrite_contexts = kwargs["overwrite_contexts"]
        overwrite_embeddings = kwargs["overwrite_embeddings"]
    else:
        overwrite_contexts = False
        overwrite_embeddings = False
    
    override = overwrite_contexts or overwrite_embeddings
    output_dir = output_dir if output_dir is not None else root_dir / config.OUTPUT_DIR
    # ===============================================================
    # Load Data
    # ===============================================================
    
    R_trn, R_val, F_trn, F_val, stories_trn, stories_val = load_data(
        subjects,
        modality,
        mode,
        stories,
        root_dir / config.FEATURE_PATH,
        root_dir / config.RESPONSE_PATH,
        root_dir / config.DATASEQ_PATH,
        root_dir / config.CONTEXTS_FILE,
        root_dir / config.EMBEDDINGS_FILE,
        verbose=verbose,
        surprisals_file = root_dir / config.FEATURE_PATH / f"surprisals.npy",
        **kwargs,
    )

    # ===============================================================
    # Prepare Data (Delays, Stacking, etc.)
    # ===============================================================
    use_keys = [mode] + (nuis_listening if modality == "listening" else nuis_reading)
    X_trn, Y_trn, X_val, Y_val, groups, story_ids = prepare_data(
        R_trn,
        R_val,
        F_trn,
        F_val,
        stories_trn,
        stories_val,
        use_keys,
        delays=np.arange(0, 9, 2),
    )

    # ===============================================================
    # Perform Ridge Regression with Cross-Validation
    # ===============================================================

    for subject in subjects:
        # Initialize or load existing results
        results = {}
        
        existing_results = load_results(
            output_dir,
            modality,
            subject
        )
        if mode in existing_results and not override:
            print(f"Results for mode '{mode}' already exist for subject '{subject}'. Skipping...")
            results = existing_results[mode]
            continue
        
        if verbose:
            print("="*60)
            print(f"Performing Group Ridge CV for {subject}")
            print("-"*60)
        try:
            results_grr = perform_group_ridge(
                X_trn,
                Y_trn[subject],
                groups,
                story_ids[subject],
                alphas=np.logspace(-10, 10, 21),
                use_keys=use_keys,
            )
        except Exception as e:
            print(f"Failed: {e}")
            print("Skipping to next subject.")
            continue
        
        r, r2, predictions = get_scores_and_prediction(results_grr, X_val, X_trn, Y_val[subject], groups, use_keys)
    
        # ===============================================================
        # Permutation Tests and FDR Correction
        # ===============================================================

        print("Performing permutation tests...", end="")
        score_names = ["r"]
        score_funcs = [correlation_score_split]
        pvalues = {}
        for score_name, score_func in zip(score_names, score_funcs):
            pvalues[score_name] = permutation_test(Y_val[subject], predictions, score_func, 2500)
        print(" ✓")

        fdr = apply_fdr_correction(pvalues)

        results = {
            "r": r,
            "r2": r2,
            # "pvalues": pvalues,
            "fdr": fdr,
        }

        # ===============================================================
        # Incremental save per subject with merging
        # ===============================================================
        try:
            existing_results[mode] = results
            save_results(
                output_dir,
                modality,
                subject,
                existing_results
            )
        except Exception as e:
            print(f"Error saving results for {subject}: {e}")

    # ===============================================================
    # Save Results
    # ===============================================================
    averaged_r = np.nanmean(results["r"][results["fdr"]["r"]["included_voxels_indices"]], axis=0)
    print(f"Averaged correlation (r) across voxels: {np.nanmean(averaged_r)}")

    # Save results to npz file
    # pu.plot_correlation_on_flatmap(
    #     example_subject, modality, mode, results, root_dir / config.MAPPER_PATH
    # )

    print("Script execution finished.")
    return results

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

        pipeline(
            args.subject,
            args.modality,
            args.mode,
            config.STORIES,
            config.NUIS_LISTENING,
            config.NUIS_READING,
            args.fdir,
            args.verbose,
            **eval(args.kwargs),
        )
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
