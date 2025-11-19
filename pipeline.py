import os
import numpy as np
from scipy.stats import zscore, pearsonr, norm
import matplotlib.pyplot as plt
import cortex

from himalaya.ridge import GroupRidgeCV
from himalaya.kernel_ridge import MultipleKernelRidgeCV, linear_kernel
from himalaya.kernel_ridge import solve_multiple_kernel_ridge_random_search
from himalaya.backend import set_backend
from himalaya.scoring import r2_score_split, correlation_score_split
from himalaya.kernel_ridge import predict_and_score_weighted_kernel_ridge

from statsmodels.stats.multitest import fdrcorrection
from sklearn.model_selection import LeaveOneGroupOut


import utils

# ----------------------------
# IO helpers
# ----------------------------
def load_responses_train(fdir, subject, modality):
    fname_trn = os.path.join(fdir, "responses", f"subject{subject}_{modality}_fmri_data_trn.hdf")
    return utils.load_data(fname_trn)  # dict: story -> (T x V)

def load_responses_val(fdir, subject, modality):
    fname_val = os.path.join(fdir, "responses", f"subject{subject}_{modality}_fmri_data_val.hdf")
    return utils.load_data(fname_val)  # dict with "story_11"

def load_features_train(fdir):
    fname_trnF = os.path.join(fdir, "features", "features_trn_NEW.hdf")
    return utils.load_data(fname_trnF)  # dict: story -> dict(space -> T x P)

def load_features_val(fdir):
    fname_valF = os.path.join(fdir, "features", "features_val_NEW.hdf")
    return utils.load_data(fname_valF)  # dict: "story_11" -> dict(space -> T x P)

# ----------------------------
# Build matrices with exact alignment across stories
# ----------------------------
def build_story_list(F_trn, required_keys):
    stories = sorted(F_trn.keys())
    # Keep only stories that contain all required keys
    ok = [s for s in stories if set(required_keys).issubset(F_trn[s].keys())]
    if not ok:
        raise ValueError("No story contains all required feature keys.")
    return ok

def stack_features(F, stories, keys, standardize=True):
    blocks = {}
    for s in stories:
        X = np.hstack([np.asarray(F[s][k]) for k in keys])
        if standardize:
            X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        blocks[s] = X
    return blocks

def stack_responses(R, stories, trim, standardize=True):
    Ys, lens = [], []
    for s in stories:
        Y = np.asarray(R[s][trim:])
        if standardize:
            Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)
        Ys.append(np.nan_to_num(Y))
        lens.append(Y.shape[0])
    return np.vstack(Ys), np.array(lens)

def delay_and_stack(X, stories, delays):
    blocks = []
    for s in stories:
        X_d = make_delayed(X[s], delays)
        blocks.append(X_d)
    return np.vstack(blocks)
    
def make_delayed(stim, delays, circpad=False):
    """Creates non-interpolated concatenated delayed versions of [stim] with the given [delays] 
    (in samples).
    
    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
    """
    nt,ndim = stim.shape
    dstims = []
    for di,d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        if d<0: ## negative delay
            dstim[:d,:] = stim[-d:,:]
            if circpad:
                dstim[d:,:] = stim[:-d,:]
        elif d>0:
            dstim[d:,:] = stim[:-d,:]
            if circpad:
                dstim[:d,:] = stim[-d:,:]
        else: ## d==0
            dstim = stim.copy()
        dstims.append(dstim)
    return np.hstack(dstims)


def align_Y_after_delays(Y_blocks, lens, lens_d):
    # Drop the first (lens - lens_d) rows per story to align with X after delays
    Ys = []
    cursor = 0
    for L, Ld in zip(lens, lens_d):
        Y_block = Y_blocks[cursor:cursor+L, :]
        drop = L - Ld
        Ys.append(Y_block[drop:, :])
        cursor += L
    return np.vstack(Ys)

def build_feature_groups(F_one_story, keys, n_delays):
    # Produce a group index per column in the delayed design
    group_idx_raw = []
    for gi, k in enumerate(keys):
        group_idx_raw.append(np.full(F_one_story[k].shape[1], gi, dtype=int))
    group_idx_raw = np.hstack(group_idx_raw)
    groups_delayed = np.repeat(group_idx_raw, n_delays)
    return groups_delayed

def make_story_ids(lens_d):
    # Sample-level group labels for LOGO
    return np.concatenate([np.full(Ld, i, dtype=int) for i, Ld in enumerate(lens_d)])

def semantic_only_prediction(X_val_d, coef, groups_delayed, semantic_group):
    # Zero-out non-semantic columns in coefficients
    coef_sem = coef.copy()
    keep = (groups_delayed == semantic_group)
    # expand keep to match delayed columns
    coef_sem[~keep, :] = 0.0
    return X_val_d @ coef_sem

def build_kernels_from_groups(X, groups_delayed):
    # Build separate kernels for each feature group
    kernels = []
    unique_groups = np.unique(groups_delayed)
    
    for group in unique_groups:
        mask = groups_delayed == group
        X_group = X[:, mask]
        kernel = X_group @ X_group.T
        kernels.append(kernel)
    
    return np.stack(kernels, axis=0)

def main(fdir, subject, modality, semantic_key, nuis_reading, nuis_listening, delays, alphas):
    R_trn = load_responses_train(fdir, subject, modality)
    R_val = load_responses_val(fdir, subject, modality)
    F_trn = load_features_train(fdir)
    F_val = load_features_val(fdir)

    # 2) Decide keys based on modality and availability
    nuis = nuis_listening if modality == "listening" else nuis_reading
    use_keys = [semantic_key] + nuis
    
    stories_ok = build_story_list(F_trn, use_keys)
    
    # 3) Stack the features per story 
    X_train = stack_features(F_trn, stories_ok, use_keys, standardize=True)
    X_train = delay_and_stack(X_train, stories_ok, delays)
    Y_train, lens = stack_responses(R_trn, stories_ok, 5, standardize=True)
    
    # 4) Group indices and story IDs for banded ridge (one alpha per feature space, expanded across delays)
    groups_delayed = build_feature_groups(F_trn[stories_ok[0]], use_keys, n_delays=len(delays))
    story_ids = make_story_ids(lens)
    
    # 5) Multiple Kernel ridge with LeaveOneGroupOut CV
    Xs_train = []   # list of (n_samples × p_group) arrays, one per feature space after delays
    for gi, key in enumerate(use_keys):
        cols = np.where(groups_delayed == gi)[0]
        Xs_train.append(X_train[:, cols])
        
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
    
    # 6) Calculate scores
    deltas = backend.to_numpy(results[0])
    dual_weights = backend.to_numpy(results[1])
    cv_scores = backend.to_numpy(results[2])
    
    X_val = stack_features(F_val, ['story_11'], use_keys, standardize=True)
    X_val = delay_and_stack(X_val, ['story_11'], delays)
    Y_val = zscore(R_val['story_11'].mean(0)[5:])

    Ks_val = []
    for gi, key in enumerate(use_keys):
        # Columns belonging to this feature group
        cols = np.where(groups_delayed == gi)[0]
    
        # Compute linear kernel (val x train)
        K_val = X_val[:, cols] @ X_train[:, cols].T
        Ks_val.append(K_val.astype(np.float32))
    
    # Stack into shape (n_kernels, n_val, n_train)
    Ks_val = np.stack(Ks_val)
    
    # cast to GPU
    Y_val = backend.asarray(Y_val, dtype=backend.float32)
    Ks_val = backend.asarray(Ks_val, dtype=backend.float32)
    
    split = False
    score_funcs = [r2_score_split, correlation_score_split]
    score_names = ["r2", "r"]
    scores = {}
    for score_name, score_func in zip(score_names, score_funcs):
        scores[score_name] = backend.to_numpy(
            predict_and_score_weighted_kernel_ridge(
                Ks_val,
                dual_weights,
                deltas,
                Y_val,
                split=split,
                n_targets_batch=512,
                score_func=score_func,
            )
        )
    
    # 7) Per-voxel correlation
    r    = scores['r'] # chnage name to just r
    r_sq = scores['r2'] # chnage name to just r2, misleading
    
    r = np.nan_to_num(r)

    # 8) FDR-Correction
    # convert r to z-scores assuming null is r ~ 0 (this is oversimplified but works well)
    T = X_val.shape[0]
    z = r * np.sqrt(T - 3) 
    pvals = 1 - norm.cdf(z)  # one-sided test: positive correlations
    
    sig_mask, _ = fdrcorrection(pvals, alpha=0.05)
    
    r_sig = r * sig_mask    # mask out non-significant voxels

    # 8) Map to flatmap and save outputs
    
    plot_args = dict(cmap='inferno',
                    with_colorbar=True, with_labels=True,
                    with_curvature=True, recache=True,
                    curvature_contrast=0.25, curvature_brightness=0.5,
                    thick=1)
    
    map_file = os.path.join(fdir, "mappers", f"subject{subject}_mappers.hdf")
    flatmap = utils.map_to_flat(r, map_file)
    flatmap = np.nan_to_num(flatmap)
    
    os.makedirs("outputs", exist_ok=True)
    np.save("outputs/subject{}_{}_r.npy".format(subject, modality), r)
    np.save("outputs/subject{}_{}_flatmap.npy".format(subject, modality), flatmap)
    
    plt.imshow(flatmap, cmap="inferno")
    plt.axis("off")
    plt.title(f"Semantic prediction r: subject {subject}, {modality}")
    plt.savefig("outputs/subject{}_{}_flatmap.png".format(subject, modality), bbox_inches="tight", dpi=150)
    
    print("Done. Saved:")
    print(" - outputs/subject{}_{}_r.npy".format(subject, modality))
    print(" - outputs/subject{}_{}_flatmap.npy".format(subject, modality))
    print(" - outputs/subject{}_{}_flatmap.png".format(subject, modality))

if __name__ == "__main__":
    
    # ----------------------------
    # Configuration
    # ----------------------------
    backend = set_backend("torch_cuda")
    fdir = os.path.abspath("/mnt/data/ni/simonee")
    subject = "01"
    modality = "listening"  # "listening" or "reading"
    semantic_key = "english1000"
    # Dataset-specific available keys
    nuis_reading  = ["letters", "numletters", "word_length_std", "numwords", "pauses"]
    nuis_listening = ["phonemes", "numphonemes", "numwords", "pauses"]
    
    delays = list(range(5))    # 0..9 (approx 0–18 s for TR~2s)
    alphas = np.logspace(1, 8, 12)  # regularization grid

    main(fdir, subject, modality, semantic_key, nuis_reading, nuis_listening, delays, alphas)
