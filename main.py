import os
import numpy as np
import matplotlib.pyplot as plt
import cortex
import utils 
import torch

from himalaya.ridge import GroupRidgeCV
from himalaya.kernel_ridge import MultipleKernelRidgeCV, linear_kernel
from himalaya.kernel_ridge import solve_multiple_kernel_ridge_random_search, predict_and_score_weighted_kernel_ridge
from himalaya.scoring import r2_score_split, correlation_score_split
from himalaya.backend import set_backend
from sklearn.model_selection import LeaveOneGroupOut
from typing import Dict, List, Union
from scipy.stats import zscore, pearsonr, norm
from scipy.interpolate import interp1d
from statsmodels.stats.multitest import fdrcorrection
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from data_sequence import DataSequence
from textgrid_utils import load_generic_trfiles, load_textgrid_transcripts
import data_loading as dl

print("Starting the script...")

backend = set_backend("torch_cuda")
modality = 'listening'
subjects = ['01']
semantic_key = "english1000"
stories = [
    'story_01', 'story_02', 'story_03', 'story_04', 'story_05',
    'story_06', 'story_07', 'story_08', 'story_09', 'story_10', 'story_11'
]
nuis_reading  = ["letters", "numletters", "word_length_std", "numwords", "pauses"]
nuis_listening = ["phonemes", "numphonemes", "numwords", "pauses"]
trfile_dir = os.path.join('stimuli', 'textgrids', 'trfiles')
transcript_dir = os.path.join('stimuli', 'textgrids', 'transcripts')
story_names = [
    'alternateithicatom',
    'avatar',
    'howtodraw',
    'legacy',
    'life',
    'myfirstdaywiththeyankees',
    'naked',
    'odetostepfather',
    'souls',
    'undertheinfluence',
    'wheretheressmoke',
]

def pipeline(R_trn, R_val, F_trn, F_val, stimuli, label, modality, stories_trn, story_val, delays=np.arange(0, 5)):
    print(f"Running pipeline for modality: {modality}, label: {label}")
    nuis = nuis_listening if modality == "listening" else nuis_reading
    use_keys = [label] + nuis

    print("Loading stimulus word sequences...")
    wordseq = dl.load_stimulus_word_sequences(stories, stimuli, trfile_dir, transcript_dir)
    embeddings_file = os.path.join("outputs", f"{modality}_{label}_embeddings.npy")
    if os.path.exists(embeddings_file):
        print(f"Loading existing embeddings from {embeddings_file}...")
        embeddings = np.load(embeddings_file, allow_pickle=True).item()
    else:
        print("Generating contextual embeddings...")
        embeddings = dl.contextual_embeddings(wordseq, "gpt2", 8, add_special_tokens=False, avg_tokens=False, interp='mean')
        print(f"Saving embeddings to {embeddings_file}...")
        os.makedirs("outputs", exist_ok=True)
        np.save(embeddings_file, embeddings)
        
    print("Preparing training and validation features...")
    F_trn_ = dl.features_with_embeddings(F_trn, stories_trn, embeddings, label)
    F_val_ = dl.features_with_embeddings(F_val, [story_val], embeddings, label)
    
    print("Stacking and delaying features for training...")
    X_trn = dl.stack_features(F_trn_, stories_trn, use_keys, standardize=True)
    X_trn = dl.delay_features(X_trn, stories_trn, delays=delays)
    X_trn = dl.stack_stories(X_trn, stories_trn)
    
    print("Stacking and delaying features for validation...")
    X_val = dl.stack_features(F_val_, [story_val], use_keys, standardize=True)
    X_val = dl.delay_features(X_val, [story_val], delays)
    X_val = dl.stack_stories(X_val, [story_val])
    
    print("Stacking responses...")
    Y_trn, lens = dl.stack_responses(R_trn, stories_trn, 5, standardize=True)
    Y_val = zscore(R_val[story_val].mean(0)[5:])
    
    print("Building feature groups...")
    groups = dl.build_feature_groups(F_trn_[stories[0]], use_keys, n_delays=len(delays))
    story_ids = np.concatenate([np.full(Ld, i, dtype=int) for i, Ld in enumerate(lens)])
    
    print("Performing Group Ridge CV...")
    results = perform_group_ridge_cv(X_trn, Y_trn, groups, story_ids, alphas=np.logspace(5, 15, 11), use_keys=use_keys)

    print("Processing results...")
    deltas = backend.to_numpy(results[0])
    dual_weights = backend.to_numpy(results[1])
    cv_scores = backend.to_numpy(results[2])
    
    print("Computing kernels for validation...")
    Ks_val = []
    for gi, key in enumerate(use_keys):
        # Columns belonging to this feature group
        cols = np.where(groups == gi)[0]

        # Compute linear kernel (val x train)
        K_val = X_val[:, cols] @ X_trn[:, cols].T
        Ks_val.append(K_val.astype(np.float32))

    # Stack into shape (n_kernels, n_val, n_train)
    Ks_val = np.stack(Ks_val)
    Y_val = backend.asarray(Y_val, dtype=backend.float32)
    Ks_val = backend.asarray(Ks_val, dtype=backend.float32)

    print("Calculating correlations...")
    r, r2 = get_correlation(Ks_val, dual_weights, deltas, Y_val)
    
    T = X_val.shape[0]
    z = r[0] * np.sqrt(T - 3) 
    pvals = 1 - norm.cdf(z)  # one-sided test: positive correlations

    print("Applying FDR correction...")
    sig_mask, _ = fdrcorrection(pvals, alpha=0.05)

    r_sig = r[0] * sig_mask    # mask out non-significant voxels

    print("Pipeline execution complete.")
    return r[0], r2[0], r_sig

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

def perform_group_ridge_cv(X_train, Y_train, groups_delayed, story_ids, alphas, use_keys):
    Xs_train = []   # list of (n_samples × p_group) arrays, one per feature space after delays
    for gi, key in enumerate(use_keys):
        cols = np.where(groups_delayed == gi)[0]
        Xs_train.append(X_train[:, cols])

    # 7) Grouped ridge with LeaveOneGroupOut CV
    print("Fitting MultipleKernelRidgeCV model...")

    # Cast to GPU
    Xs_train = [backend.asarray(X_train, dtype="float32") for X_train in Xs_train]
    Ks_train = backend.stack([X_train @ X_train.T for X_train in Xs_train])
    Ks_train = backend.asarray(Ks_train, dtype=backend.float32)
    Y_train = backend.asarray(Y_train, dtype=backend.float32)

    cv_splits = list(LeaveOneGroupOut().split(np.zeros(Ks_train.shape[1]), groups=story_ids))

    print(f"Created {len(cv_splits)} CV splits.")

    results = solve_multiple_kernel_ridge_random_search(
            Ks=Ks_train,
            Y=Y_train,
            alphas=alphas,
            n_targets_batch=512,
            return_weights='dual',
            n_alphas_batch=20,
            n_targets_batch_refit=200,
            jitter_alphas=True,
        )

    print("Model fitting complete.")
    return results

def plot_correlation_on_flatmap(subject: str, modality: str, semantic_key: str, r: np.ndarray, r_sig: np.ndarray, save_fig: bool = True, fdir: str = "."):
    cmap_ = plt.cm.hot 

    # 2. Set the color for 'bad' (NaN/masked) values to Grey
    cmap_.set_bad(color='white') 
    cmap_.set_under(color='#555555')

    map_file = os.path.join(fdir, "mappers", f"{subject}_mappers.hdf")
    flatmap_sig = utils.map_to_flat(r_sig, map_file)
    flatmap = utils.map_to_flat(r, map_file)
    #flatmap = np.nan_to_num(flatmap)

    os.makedirs("outputs", exist_ok=True)
    np.save("outputs/subject{}_{}_r.npy".format(subject, modality), r)
    np.save("outputs/subject{}_{}_flatmap.npy".format(subject, modality), flatmap)

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # Plot significant r values
    axes[0].imshow(flatmap_sig, cmap=cmap_, vmin=0, vmax=0.4)
    axes[0].axis("off")
    axes[0].set_title(f"Significant r values: subject {subject}, {modality}")

    # Plot normal r values
    axes[1].imshow(flatmap, cmap=cmap_, vmin=0, vmax=0.4)
    axes[1].axis("off")
    axes[1].set_title(f"All r values: subject {subject}, {modality}")

    plt.tight_layout()
    if save_fig:
        print("Saving flatmap figure to outputs/{}_{}_{}_flatmap.png".format(subject, modality, semantic_key))
        plt.savefig("outputs/{}_{}_{}_flatmap.png".format(subject, modality, semantic_key), bbox_inches="tight", dpi=150)
    plt.show()
    
def main(subject, modality, semantic_key, fdir):
    print("Loading responses...")
    R_trn = dl.load_responses([subject], modality, split='trn', fdir=fname)
    R_val = dl.load_responses([subject], modality, split='val', fdir=fname)

    print("Loading features...")
    F_trn = dl.load_features(split='trn')
    F_val = dl.load_features(split='val')
    
    print(f"Running pipeline for subject {subject}...")
    r, r2, r_sig = pipeline(
        R_trn[subject],
        R_val[subject],
        F_trn,
        F_val,
        stimuli=story_names,
        label=semantic_key,
        modality=modality,
        stories_trn=stories[:10],
        story_val=stories[10]
    )
    print("Plotting results...")
    plot_correlation_on_flatmap(subject, modality, semantic_key, r, r_sig)
    print("Script execution finished.")

if __name__ == "__main__":
    subject ="subject07"
    fname = './'
    modality = 'listening'
    semantic_key = "nonsense"
    main(subject, modality, semantic_key, fname)
