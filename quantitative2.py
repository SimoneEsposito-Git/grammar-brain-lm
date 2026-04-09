import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.sparse
import pickle
import nibabel as nib
from data_loading.file_io import load_results
from data_loading import config

language_rois = ['IFG', 'IFGOrb', 'MFG', 'PTL', 'ATL', 'AG', 'PCC', 'dmPFC']

def load_sparse_array(fname, varname):
    """Load a numpy sparse array from an hdf file"""
    with h5py.File(fname) as hf:
        data = (
            hf["%s_data" % varname],
            hf["%s_indices" % varname],
            hf["%s_indptr" % varname],
        )
        sparsemat = scipy.sparse.csr_matrix(data, shape=hf["%s_shape" % varname])
    return sparsemat 

def get_significant_voxels(results, mode):
    sig = np.array(results[mode]['r'])
    sig[results[mode]['fdr']['r']['excluded_voxels_indices']] = np.nan
    return sig

def map_data_to_fsaverage(voxels: np.ndarray, map_file: str) -> np.ndarray:
    """Convert voxel data to flatmap space"""
    pixmap = load_sparse_array(map_file, "voxel_to_fsaverage")
    badmask = np.array(pixmap.sum(1) > 0).ravel()
    mimg = (np.nan * np.ones(badmask.shape)).astype(voxels.dtype)
    mimg[badmask] = (pixmap * voxels.ravel())[badmask].astype(mimg.dtype)
    return mimg

def get_language_mask(roi: str, return_fs7: bool = True) -> np.ndarray:
    assert roi in language_rois, f"ROI {roi} is not in the predefined language ROIs: {language_rois}"
    roi_indices = {'AG': [149, 150, 151, 140, 141],
                    'ATL': [123, 131, 132, 134, 172, 128, 176],
                    'PTL': [129, 125, 139, 25, 28, 175, 130],
                    'IFG': [74, 75, 79, 81],
                    'MFG': [12],
                    'IFGOrb': [77, 171, 85],
                    'PCC': [161, 162, 27, 30, 32, 33, 34, 14],
                    'dmPFC': [69, 62, 72]}

    if return_fs7:
        language_mask = []
        path_to_lh_annot = config.DATA_DIR/'fsaverage'/'lh.HCP-MMP1.annot'
        path_to_rh_annot = config.DATA_DIR/'fsaverage'/'rh.HCP-MMP1.annot'
        lh_annot = nib.freesurfer.io.read_annot(path_to_lh_annot)
        rh_annot = nib.freesurfer.io.read_annot(path_to_rh_annot)
        language_mask = np.zeros(327684, dtype=bool)
        # if roi == 'lh':
        #     language_mask[:163842] = 1
        #     return language_mask
        # if roi == 'rh':
        #     language_mask[163842:] = 1
        #     return language_mask
        
        for index in roi_indices[roi]:
            language_mask[np.where(lh_annot[0] == index)[0]] = 1
            language_mask[163842+np.where(rh_annot[0] == index)[0]] = 1
        return language_mask
    
    path_to_lh = config.DATA_DIR/'fsaverage'/'tpl-fsaverage6_hemi-L_desc-MMP_dseg.label.gii'
    path_to_rh = config.DATA_DIR/'fsaverage'/'tpl-fsaverage6_hemi-R_desc-MMP_dseg.label.gii'

    lh = nib.load(path_to_lh)
    rh = nib.load(path_to_rh)

    language_mask = np.zeros(327684, dtype=bool)
    for index in roi_indices[roi]:
        language_mask[np.where(lh.darrays[0].data == index)[0]] = 1
        language_mask[163842+np.where(rh.darrays[0].data == index)[0]] = 1

    return language_mask

def compute_delta(noun: np.ndarray, verb: np.ndarray, baseline: np.ndarray, mask: np.ndarray) -> tuple[float, float, int, int]:
    """
    Compute mean positive delta for noun and verb sensitivity within a masked ROI.
    dnoun = mean(baseline - noun), positive values only
    dverb = mean(baseline - verb), positive values only
    Returns (dnoun_mean, dverb_mean, n_valid, roi_size)
    """
    dnoun = baseline[mask] - noun[mask]
    dverb = baseline[mask] - verb[mask]

    roi_size = int(np.sum(mask))

    dnoun_pos = dnoun[dnoun > 0]
    dverb_pos = dverb[dverb > 0]

    n_valid = max(len(dnoun_pos), len(dverb_pos))

    dnoun_mean = float(np.nanmean(dnoun_pos)) if len(dnoun_pos) > 0 else np.nan
    dverb_mean = float(np.nanmean(dverb_pos)) if len(dverb_pos) > 0 else np.nan

    return dnoun_mean, dverb_mean, n_valid, roi_size

def plot_2d_dominance(stats_all: dict, n_subjects: int,
                      roi_groups: dict | None = None,
                      output_dir: str | None = None,
                      show_plots: bool = False):
    """
    Horizontal dot plot with two dots per ROI per modality:
    - Blue dot + error bar: dnoun (mean positive baseline - noun across subjects)
    - Orange dot + error bar: dverb (mean positive baseline - verb across subjects)
    X-axis: delta r value. Y-axis: ROI name.
    Layout: Listening (left) | Labels (center) | Reading (right)
    """
    noun_color = "#4878C8"
    verb_color = "#E07A30"

    combined_rois = set()
    for mod_stats in stats_all.values():
        combined_rois.update(mod_stats.keys())

    # ── Build grouped ROI order ──────────────────────────────────────────────
    if roi_groups:
        assigned = {roi for rois in roi_groups.values() for roi in rois}
        leftover = [r for r in combined_rois if r not in assigned]
        groups_to_plot = dict(roi_groups)
        if leftover:
            groups_to_plot["Other"] = leftover
    else:
        groups_to_plot = {"": list(combined_rois)}

    ordered_rois  = []
    group_headers = {}

    for group_label, members in groups_to_plot.items():
        present = [r for r in members if r in combined_rois]
        if not present:
            continue
        if ordered_rois:
            ordered_rois.append(None)
        group_headers[len(ordered_rois)] = group_label
        ordered_rois.extend(present)

    # ── Y positions ──────────────────────────────────────────────────────────
    GAP     = 0.5
    SPACING = 0.7   # slightly more room for two dots per row

    y_positions = {}
    y           = 0.0
    y_ticks     = []
    y_dividers  = []

    for entry in ordered_rois:
        if entry is None:
            y_dividers.append(y - GAP / 2)
            y += GAP
        else:
            y_positions[entry] = y
            y_ticks.append((y, entry))
            y += SPACING

    header_ys = {}
    for sentinel_idx, label in group_headers.items():
        for j in range(sentinel_idx, len(ordered_rois)):
            if ordered_rois[j] is not None:
                header_ys[label] = y_positions[ordered_rois[j]]
                break

    total_height = y

    # ── Layout ───────────────────────────────────────────────────────────────
    fig_height = max(5, total_height * 0.38 + 1)
    wspace     = 0.3

    fig, (ax_list, ax_read) = plt.subplots(1, 2, figsize=(4.7, fig_height))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.16, top=0.92, wspace=wspace)

    listen_key = next((k for k in stats_all if 'listen' in k.lower()), None)
    read_key   = next((k for k in stats_all if 'read'   in k.lower()), None)

    axes_info = [
        (ax_list, listen_key, 'Listening'),
        (ax_read, read_key,   'Reading'),
    ]

    OFFSET = 0.12   # vertical jitter so noun/verb dots don't overlap

    for ax, mod, title in axes_info:
        roi_stats = stats_all.get(mod, {})

        #ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)
        
        for yd in y_dividers:
            ax.axhline(yd, color="lightgray", linewidth=0.8, zorder=1)

        for roi, ypos in y_positions.items():
            if roi not in roi_stats:
                continue
            entry = roi_stats[roi]

            dnoun_mean = entry["dnoun_mean"]
            dnoun_std  = entry["dnoun_std"]
            dverb_mean = entry["dverb_mean"]
            dverb_std  = entry["dverb_std"]

            if not np.isnan(dnoun_mean):
                ax.errorbar(dnoun_mean, ypos - OFFSET,
                            xerr=dnoun_std,
                            fmt="o", color=noun_color, ecolor=noun_color,
                            elinewidth=1.2, capsize=3, capthick=1.2,
                            markersize=5, zorder=2)

            if not np.isnan(dverb_mean):
                ax.errorbar(dverb_mean, ypos + OFFSET,
                            xerr=dverb_std,
                            fmt="o", color=verb_color, ecolor=verb_color,
                            elinewidth=1.2, capsize=3, capthick=1.2,
                            markersize=5, zorder=2)

        ax.set_ylim(-0.8, total_height - 0.2)
        ax.set_xlim(0, 0.08)
        ax.set_xticks([0, 0.02, 0.04, 0.06, 0.08])
        ax.invert_yaxis()
        ax.set_xlabel("Δr", fontsize=8)
        ax.set_title(title, fontsize=12, pad=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.xaxis.grid(True, linewidth=0.5, color="lightgray", zorder=0)
        ax.set_axisbelow(True)

    # ── Formatting Listening (Left) ──────────────────────────────────────────
    ax_list.spines["top"].set_visible(False)
    ax_list.spines["left"].set_visible(True)
    ax_list.spines["right"].set_visible(False)
    ax_list.spines["bottom"].set_visible(True)
    ax_list.set_yticks([y for y, _ in y_ticks])
    ax_list.set_yticklabels([])
    ax_list.tick_params(axis="y", length=0)

    # ── Formatting Reading (Right) ───────────────────────────────────────────
    ax_read.spines["top"].set_visible(False)
    ax_read.spines["left"].set_visible(True)
    ax_read.spines["right"].set_visible(False)
    ax_read.spines["bottom"].set_visible(True)
    ax_read.set_yticks([])
    ax_read.tick_params(axis="y", length=0)

    # ── Centre ROI labels ────────────────────────────────────────────────────
    center_x = 1.0 + (wspace / 2)
    for y, label in y_ticks:
        ax_list.text(center_x, y, label,
                     transform=ax_list.get_yaxis_transform(),
                     fontsize=8, ha="center", va="center")

    for label, yh in header_ys.items():
        ax_list.text(0.02, yh - GAP + 0.1, label,
                     transform=ax_list.get_yaxis_transform(),
                     fontsize=8, fontweight="bold", color="gray",
                     ha="left", va="bottom")

    # ── Legend ───────────────────────────────────────────────────────────────
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=noun_color,
                   markersize=7, label="Noun-masked"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=verb_color,
                   markersize=7, label="Verb-masked"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=8,
               loc="lower center", bbox_to_anchor=(0.5, 0), ncol=3)

    if output_dir:
        plt.savefig(f"{output_dir}/roi_dominance_2d.pdf", bbox_inches="tight", pad_inches=0.15)
    if show_plots:
        plt.show()
       
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quantitative analysis of flatmaps")
    parser.add_argument("--subjects", nargs="+", default=config.SUBJECTS, help="List of subjects")
    parser.add_argument("--modalities", nargs="+", default=config.MODALITIES, help="List of modalities")
    parser.add_argument("--rois", nargs="+", default=language_rois, help="List of ROIs to analyze")
    parser.add_argument("--output_dir", default=config.OUTPUT_DIR/'figures', help="Directory to save output figures")
    parser.add_argument("--min_voxels", type=int, default=10, help="Absolute floor for statistical stability")
    parser.add_argument("--min_roi_pct", type=float, default=0.10, help="Minimum proportion of the ROI that must be valid")
    parser.add_argument("--show_plots", action="store_true", help="Whether to display plots interactively")
    args = parser.parse_args()

    stats_all = {}

    for modality in args.modalities:
        roi_subject_values = {roi: [] for roi in args.rois}

        for subject in args.subjects:
            try:
                results = load_results(
                    config.OUTPUT_DIR / "results_actual",
                    modality,
                    subject,
                )
                verbs    = get_significant_voxels(results, "verb")
                nouns    = get_significant_voxels(results, "noun")
                baseline = get_significant_voxels(results, "baseline")

                mapper = config.MAPPER_PATH / f"{subject}_mappers.hdf"
                verbs    = map_data_to_fsaverage(verbs,    mapper)
                nouns    = map_data_to_fsaverage(nouns,    mapper)
                baseline = map_data_to_fsaverage(baseline, mapper)
                print(f"Loaded and mapped {subject} ({baseline.shape[0]} voxels).")
            except Exception as e:
                print(f"Skipping {subject} in {modality}: {e}")
                continue

            for roi in args.rois:
                try:
                    mask = get_language_mask(roi)
                except KeyError:
                    print(f"ROI {roi} not found for {subject}; skipping.")
                    continue

                dnoun_mean, dverb_mean, n_valid, roi_size = compute_delta(nouns, verbs, baseline, mask)

                valid_pct = n_valid / roi_size if roi_size > 0 else 0
                if n_valid < args.min_voxels or valid_pct < args.min_roi_pct:
                    dnoun_mean, dverb_mean = np.nan, np.nan

                roi_subject_values[roi].append((dnoun_mean, dverb_mean))

        roi_stats = {}
        for roi in args.rois:
            values = np.array(roi_subject_values[roi], dtype=float)  # (n_subjects, 2)

            if values.size == 0 or np.all(np.isnan(values)):
                print(f"No valid data for ROI {roi} in {modality}.")
                roi_stats[roi] = {
                    "dnoun_mean": np.nan, "dnoun_std": np.nan,
                    "dverb_mean": np.nan, "dverb_std": np.nan,
                }
                continue

            values = values[~np.all(np.isnan(values), axis=1)]
            if values.size == 0:
                print(f"All subjects below threshold for ROI {roi} in {modality}.")
                roi_stats[roi] = {
                    "dnoun_mean": np.nan, "dnoun_std": np.nan,
                    "dverb_mean": np.nan, "dverb_std": np.nan,
                }
                continue

            roi_stats[roi] = {
                "dnoun_mean": np.nanmean(values[:, 0]),
                "dnoun_std":  np.nanstd(values[:, 0]),
                "dverb_mean": np.nanmean(values[:, 1]),
                "dverb_std":  np.nanstd(values[:, 1]),
            }
            print(f"  {modality} | {roi}: dnoun={roi_stats[roi]['dnoun_mean']:.4f} ± {roi_stats[roi]['dnoun_std']:.4f}, "
                  f"dverb={roi_stats[roi]['dverb_mean']:.4f} ± {roi_stats[roi]['dverb_std']:.4f}")

        stats_all[modality] = roi_stats

    plot_2d_dominance(
        stats_all,
        n_subjects=len(args.subjects),
        roi_groups=None,
        output_dir=args.output_dir,
        show_plots=args.show_plots,
    )