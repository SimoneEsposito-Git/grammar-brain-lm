import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.sparse
import pickle
import nibabel as nib
from data_loading.file_io import load_results
from data_loading import config

language_rois = ['AG', 'ATL', 'PTL', 'IFG', 'MFG', 'IFGOrb', 'PCC', 'dmPFC']

# ==== Utility functions for loading and processing voxel data ====
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

def get_voxels(results, mode):
    voxels = np.array(results[mode]['r'])
    return voxels

# ==== Individual voxel mapping functions ====
def map_to_flat(voxels: np.ndarray, map_file: str) -> np.ndarray:
    """Convert voxel data to flatmap space"""
    pixmap = load_sparse_array(map_file, "voxel_to_flatmap")
    with h5py.File(map_file, mode="r") as hf:
        pixmask = hf["flatmap_mask"][()]
    badmask = np.array(pixmap.sum(1) > 0).ravel()
    img = (np.nan * np.ones(pixmask.shape)).astype(voxels.dtype)
    mimg = (np.nan * np.ones(badmask.shape)).astype(voxels.dtype)
    mimg[badmask] = (pixmap * voxels.ravel())[badmask].astype(mimg.dtype)
    img[pixmask] = mimg
    return img.T[::-1]

def load_roi_mask(mask_file: str, roi) -> np.ndarray:
    """Load ROI mask from file"""
    with h5py.File(mask_file, mode="r") as hf:
        roi_entry = hf[f"roi_mask_{roi}"]
        if isinstance(roi_entry, h5py.Group):
            raise KeyError(f"ROI {roi} points to an HDF group, not a dataset.")
        mask = roi_entry[()]
    mask_ = np.asarray(mask, dtype=bool)
    return mask_.T[::-1]

def percentage(noun: np.ndarray, verb: np.ndarray, baseline: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    noun_masked = baseline[mask] - noun[mask] 
    verb_masked = baseline[mask] - verb[mask] 

    roi_size = np.sum(mask)

    valid_voxels = np.isfinite(noun_masked) | np.isfinite(verb_masked)
    n_valid = np.sum(valid_voxels)
    
    if n_valid == 0:
        return np.array([np.nan, np.nan], dtype=float), 0, roi_size

    stacked = np.stack([noun_masked, verb_masked], axis=0)
    winner_indices = np.argmax(stacked, axis=0)[valid_voxels]
    counts = np.bincount(winner_indices.flatten(), minlength=stacked.shape[0])
    percentages = counts / winner_indices.size * 100
    
    return percentages, n_valid, roi_size

# ==== FSAverage mapping and language mask functions ====
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

def percentage2(noun: np.ndarray, verb: np.ndarray, baseline: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    noun_masked = noun[mask] 
    verb_masked = verb[mask] 

    roi_size = np.sum(mask)
    
    noun_count = np.nansum(noun_masked)
    verb_count = np.nansum(verb_masked)
    total_count = noun_count + verb_count
    
    percentages = np.array([noun_count/total_count * 100, verb_count/total_count * 100])
    return percentages, total_count, roi_size

# ==== Plotting function for paired noun/verb dominance ====
def plot_paired_percentage(stats_all: dict, n_subjects: int,
                           roi_groups: dict | None = None, output_dir: str | None = None, 
                           show_plots: bool = False):
    """Plot paired noun/verb dominance as a horizontal diverging dot plot.
    Arrangement: Listening | Labels | Reading
    """
    if not stats_all:
        print("No valid ROI data found.")
        return

    noun_color = "#4878C8"
    verb_color = "#E07A30"

    combined_rois = set()
    for mod_stats in stats_all.values():
        combined_rois.update(mod_stats.keys())

    # ── Build grouped ROI order ──────────────────────────────────────────────
    if roi_groups:
        assigned = {roi for rois in roi_groups.values() for roi in rois}
        leftover  = [r for r in combined_rois if r not in assigned]
        groups_to_plot = dict(roi_groups)
        if leftover:
            groups_to_plot["Other"] = leftover
    else:
        groups_to_plot = {"": list(combined_rois)}

    ordered_rois   = []   
    group_headers  = {}   

    for group_label, members in groups_to_plot.items():
        present = [r for r in members if r in combined_rois]
        if not present:
            continue

        if ordered_rois:               
            ordered_rois.append(None)

        group_headers[len(ordered_rois)] = group_label
        ordered_rois.extend(present)

    # ── Compute plot values ──────────────────────────────────────────────────
    def get_vals(roi_stats, roi):
        nm, vm = roi_stats[roi]["mean"]
        ns, vs = roi_stats[roi]["std"]
        noun_dom = nm >= vm
        plot_mean = nm if noun_dom else 100 - vm   
        plot_std  = ns if noun_dom else vs
        color     = noun_color if noun_dom else verb_color
        return plot_mean, plot_std, color

    # ── Layout ───────────────────────────────────────────────────────────────
    GAP = 0.5   
    SPACING = 0.5
    wspace = 0.3

    y_positions = {}   
    y          = 0.0
    y_ticks    = []    
    y_dividers = []    

    for entry in ordered_rois:
        if entry is None:
            y_dividers.append(y - GAP/2)  
            y += GAP
        else:
            if len(ordered_rois) > 1:
                idx = ordered_rois.index(entry)  
                if idx in group_headers or (idx > 0 and ordered_rois[idx - 1] is None and
                                             ordered_rois.index(entry) in
                                             {i + 1 for i in range(len(ordered_rois))
                                              if ordered_rois[i] is None}):
                    pass  
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

    # ── Draw ─────────────────────────────────────────────────────────────────
    fig_height = max(5, total_height * 0.38 + 1)
    
    fig, (ax_list, ax_read) = plt.subplots(1, 2, figsize=(4.7, fig_height))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.16, top=0.92, wspace=wspace)

    listen_key = next((k for k in stats_all.keys() if 'listen' in k.lower()), 'listening')
    read_key   = next((k for k in stats_all.keys() if 'read' in k.lower()), 'reading')

    axes_info = [
        (ax_list, listen_key, 'Listening'),
        (ax_read, read_key, 'Reading')
    ]

    for ax, mod, title in axes_info:
        roi_stats = stats_all.get(mod, {})
        
        ax.axvspan(50, 100, color=noun_color, alpha=0.04, zorder=0)
        ax.axvspan(0,  50,  color=verb_color, alpha=0.04, zorder=0)
        ax.axvline(50, color="gray", linewidth=0.8, linestyle="--", zorder=1)

        for yd in y_dividers:
            ax.axhline(yd, color="lightgray", linewidth=0.8, linestyle="-", zorder=1)

        for roi, ypos in y_positions.items():
            if roi in roi_stats and not np.all(np.isnan(roi_stats[roi]["mean"])):
                mean, std, color = get_vals(roi_stats, roi)
                ax.errorbar(mean, ypos, xerr=std, fmt="o", color=color, ecolor=color,
                            elinewidth=1.2, capsize=3, capthick=1.2, markersize=5, zorder=2)

        ax.set_xlim(0, 100)
        ax.set_ylim(-0.8, total_height - 0.2)
        ax.invert_yaxis()
        
        # Explicit ticks to align the edges and prevent text from getting cropped outside the 0.0-1.0 figure bounds
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0%", "50%", "100%"])
        
        if ax == ax_list:
            ax.get_xticklabels()[0].set_horizontalalignment("left")
        elif ax == ax_read:
            ax.get_xticklabels()[-1].set_horizontalalignment("right")

        ax.set_xlabel("Winning Voxels (%)", fontsize=8)
        ax.set_title(title, fontsize=12, pad=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.xaxis.grid(True, linewidth=0.5, color="lightgray", zorder=0)
        ax.set_axisbelow(True)

    # ── Formatting Listening (Left) ──
    ax_list.spines["top"].set_visible(False)
    ax_list.spines["left"].set_visible(False)
    ax_list.spines["right"].set_visible(True)
    ax_list.spines["bottom"].set_visible(True)
    
    ax_list.set_yticks([y for y, _ in y_ticks])
    ax_list.set_yticklabels([]) 
    ax_list.tick_params(axis="y", length=0)

    # ── Formatting Reading (Right) ──
    ax_read.spines["top"].set_visible(False)
    ax_read.spines["right"].set_visible(False)
    ax_read.spines["left"].set_visible(True)
    ax_read.spines["bottom"].set_visible(True)

    ax_read.set_yticks([])
    ax_read.tick_params(axis="y", length=0)

    # ── Inside & Middle Labels ──
    center_x = 1.0 + (wspace / 2)  # Middle of the wspace gap
    for y, label in y_ticks:
        ax_list.text(center_x, y, label, transform=ax_list.get_yaxis_transform(),
                     fontsize=8, ha="center", va="center")

    for label, yh in header_ys.items():
        # x=0.02 draws the group name just inside the left boundary of the Listening plot
        ax_list.text(0.02, yh - GAP + 0.1, label, transform=ax_list.get_yaxis_transform(),
                     fontsize=8, fontweight="bold", color="gray", ha="left", va="bottom")

    # ── Legend ──
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=verb_color, markersize=7, label="Verb sensitive"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=noun_color, markersize=7, label="Noun sensitive"),
        plt.Line2D([0], [0], color="gray", linewidth=0.8, linestyle="--", label="50% (chance)"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 0), ncol=3)

    if output_dir:
        # tight_layout will override absolute spacing bounds and create automatic margins, 
        # so it has been intentionally removed to maintain the 4.7in restriction.
        plt.savefig(f"{output_dir}/roi_dominance_paired_two.pdf")
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
            results = load_results(
                config.OUTPUT_DIR / "results_actual",
                modality,
                subject,
            )

            try:
                verbs = get_significant_voxels(results, "verb")
                nouns = get_significant_voxels(results, "noun")
                baseline = get_significant_voxels(results, "baseline")
                
                verbs = map_data_to_fsaverage(verbs, config.MAPPER_PATH / f"{subject}_mappers.hdf")
                nouns = map_data_to_fsaverage(nouns, config.MAPPER_PATH / f"{subject}_mappers.hdf")
                baseline = map_data_to_fsaverage(baseline, config.MAPPER_PATH / f"{subject}_mappers.hdf")
                print(f"Loaded and mapped data for subject {subject} with {baseline.shape[0]} voxels.")
            except Exception as e:
                print(f"Error occurred while processing subject {subject}: {e}")
                continue

            for roi in args.rois:
                try:
                    #mask = load_roi_mask(mapper, roi)
                    mask = get_language_mask(roi)
                    #print(f"Loaded mask for ROI {roi} with {np.sum(mask)} voxels.")
                    
                except KeyError:
                    print(f"ROI {roi} not found for {subject}; skipping.")
                    continue

                percentages, n_valid, roi_size = percentage2(nouns, verbs, baseline, mask)
                
                valid_pct = n_valid / roi_size if roi_size > 0 else 0
                
                if n_valid < args.min_voxels or valid_pct < args.min_roi_pct:
                    percentages = np.array([np.nan, np.nan], dtype=float)
                    
                roi_subject_values[roi].append(percentages)

        roi_stats = {}
        for roi in args.rois:
            values = np.array(roi_subject_values[roi], dtype=float)
            if values.size == 0:
                print(f"No valid data collected for ROI {roi} in modality {modality}.")
                roi_stats[roi] = {"mean": np.array([np.nan, np.nan]), "std": np.array([np.nan, np.nan])}
                continue

            values = values[~np.all(np.isnan(values), axis=1)]
            if values.size == 0:
                print(
                    f"Skipping ROI {roi} in modality {modality}: All subjects were under the threshold."
                )
                roi_stats[roi] = {"mean": np.array([np.nan, np.nan]), "std": np.array([np.nan, np.nan])}
                continue

            roi_stats[roi] = {
                "mean": np.nanmean(values, axis=0),
                "std": np.nanstd(values, axis=0),
            }
            
        stats_all[modality] = roi_stats

    plot_paired_percentage(stats_all, len(args.subjects), config.ROI_GROUPS_, output_dir=args.output_dir, show_plots=args.show_plots)