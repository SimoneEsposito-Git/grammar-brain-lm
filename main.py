import os
import numpy as np
import matplotlib.pyplot as plt
import cortex
import utils 
import torch

from himalaya.ridge import GroupRidgeCV
from himalaya.kernel_ridge import MultipleKernelRidgeCV, linear_kernel
from himalaya.kernel_ridge import solve_multiple_kernel_ridge_random_search
from himalaya.backend import set_backend
from sklearn.model_selection import LeaveOneGroupOut
from typing import Dict, List, Union
from scipy.stats import zscore, pearsonr, norm
from scipy.interpolate import interp1d
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from data_sequence import DataSequence
from textgrid_utils import load_generic_trfiles, load_textgrid_transcripts
import data_loading as dl

modality = 'listening'
subjects = ['01']
semantic_key = "english1000"
stories = [
    'story01', 'story02', 'story03', 'story04', 'story05',
    'story06', 'story07', 'story08', 'story09', 'story10', 'story11'
]
nuis_reading  = ["letters", "numletters", "word_length_std", "numwords", "pauses"]
nuis_listening = ["phonemes", "numphonemes", "numwords", "pauses"]
fdir = './'
trfile_dir = os.path.join('stimuli', 'textgrids', 'trfiles')
transcript_dir = os.path.join('stimuli', 'textgrids', 'transcripts')

def pipeline(R_trn, R_val, F_trn, F_val, stimuli, label, modality, stories_trn, stories_val):
    nuis = nuis_listening if modality == "listening" else nuis_reading
    use_keys = [label] + nuis

    wordseq = dl.load_stimulus_word_sequences(stories, stimuli, trfile_dir, transcript_dir)
    embeddings = dl.contextual_embeddings(wordseq, "gpt2", 8, add_special_tokens=False, avg_tokens=False)
    
    F_trn_ = dl.features_with_embeddings(F_trn, stories_trn, embeddings, label)
    F_val_ = dl.features_with_embeddings(F_val, stories_val, embeddings, label)

    X_trn = dl.stack_features(F_trn_, stories_trn, use_keys, standardize=True)
    X_trn = dl.delay_features(X_trn_st, stories_trn, delays=np.arange(0, 10))
    X_trn = dl.stack_stories(X_trn_d, stories_trn)
    
    X_val = dl.stack_features(F_val, stories_val, use_keys, standardize=True)
    X_val = dl.delay_features(X_val, stories_val, delays)
    X_val = dl.stack_stories(X_val, [stories_val])
    
    Y_trn, lens = dl.stack_responses(R_trn, stories_trn, 5, standardize=True)
    Y_val = zscore(R_val[stories_val].mean(0)[5:])
    
    groups = dl.build_feature_groups(F_trn_[stories[0]], use_keys, delays=np.arange(0, 10))
    story_ids = np.concatenate([np.full(Ld, i, dtype=int) for i, Ld in enumerate(lens)])
    
    results = perform_group_ridge_cv(X_trn, Y_val_st, groups, story_ids, alphas=np.logspace(5, 15, 11), use_keys=use_keys)

    deltas = backend.to_numpy(results[0])
    dual_weights = backend.to_numpy(results[1])
    cv_scores = backend.to_numpy(results[2])
    
    Ks_val = []
    for gi, key in enumerate(use_keys):
        # Columns belonging to this feature group
        cols = np.where(groups_delayed == gi)[0]

        # Compute linear kernel (val x train)
        K_val = X_val[:, cols] @ X_train[:, cols].T
        Ks_val.append(K_val.astype(np.float32))

    # Stack into shape (n_kernels, n_val, n_train)
    Ks_val = np.stack(Ks_val)
    Y_val = backend.asarray(Y_val, dtype=backend.float32)
    Ks_val = backend.asarray(Ks_val, dtype=backend.float32)

    r, r2 = get_correlation(Ks_val, dual_weights, deltas, Y_val)
    
    T = X_val.shape[0]
    z = r[0] * np.sqrt(T - 3) 
    pvals = 1 - norm.cdf(z)  # one-sided test: positive correlations

    sig_mask, _ = fdrcorrection(pvals, alpha=0.05)

    r_sig = r[0] * sig_mask    # mask out non-significant voxels

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

if __name__ == "__main__":
    R_trn = dl.load_responses(subjects, modality, split='trn')
    R_val = dl.load_responses(subjects, modality, split='val')

    F_trn = dl.load_features(split='trn')
    F_val = dl.load_features(split='val')
