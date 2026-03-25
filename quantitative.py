import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.sparse
from data_loading.file_io import load_results
from data_loading import config

def load_sparse_array(fname, varname):
    """Load a numpy sparse array from an hdf file
    
    By Mark Lescroart
    """
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

def percentage(noun: np.ndarray, verb: np.ndarray, mask: np.ndarray, mapper: str) -> tuple[np.ndarray, int, int]:
    noun_masked = noun[mask]
    verb_masked = verb[mask]

    roi_size = np.sum(mask)

    # Only compare voxels where both noun and verb are valid numbers.
    valid_voxels = np.isfinite(noun_masked) & np.isfinite(verb_masked)
    n_valid = np.sum(valid_voxels)
    
    if n_valid == 0:
        return np.array([np.nan, np.nan], dtype=float), 0, roi_size

    stacked = np.stack([noun_masked[valid_voxels], verb_masked[valid_voxels]], axis=0)
    winner_indices = np.argmax(stacked, axis=0)
    # Count how many times each flatmap wins
    counts = np.bincount(winner_indices.flatten(), minlength=stacked.shape[0])
    # Calculate percentage for each flatmap
    percentages = counts / winner_indices.size * 100
    
    return percentages, n_valid, roi_size

def plot_percentage(roi_stats: dict, modality: str, n_subjects: int,
                    roi_groups: dict | None = None, output_dir: str | None = None):
    """Plot noun/verb dominance as a horizontal diverging dot plot with ROI groups.

    Each ROI is a single row. Points left of 50% = verb dominant,
    right of 50% = noun dominant. Error bars show ± SD.
    ROIs are sorted by dominance strength within each group.

    Args:
        roi_stats:   Dictionary mapping ROI names to {"mean": [noun, verb], "std": [noun, verb]}
        modality:    Modality name
        n_subjects:  Number of subjects used in aggregation
        roi_groups:  Optional dict mapping group label -> list of ROI names.
                     ROIs not found in any group are collected in "Other".
                     If None, falls back to a flat sorted plot.
    """
    if not roi_stats:
        print(f"No valid ROI data found for modality '{modality}'.")
        return

    noun_color = "#4878C8"
    verb_color = "#E07A30"

    # ── Build grouped ROI order ──────────────────────────────────────────────
    if roi_groups:
        assigned = {roi for rois in roi_groups.values() for roi in rois}
        leftover  = [r for r in roi_stats if r not in assigned]
        groups_to_plot = dict(roi_groups)
        if leftover:
            groups_to_plot["Other"] = leftover
    else:
        groups_to_plot = {"": list(roi_stats.keys())}

    # Build flat ordered list, inserting None as a sentinel for group breaks
    ordered_rois   = []   # str | None
    group_headers  = {}   # y-index -> group label

    for group_label, members in groups_to_plot.items():
        present = [r for r in members if r in roi_stats]
        if not present:
            continue

        # Sort within group by dominance strength (weakest → strongest)
        dom = [max(roi_stats[r]["mean"]) for r in present]
        present = [present[i] for i in np.argsort(dom)]

        if ordered_rois:               # divider before every group except first
            ordered_rois.append(None)

        group_headers[len(ordered_rois)] = group_label
        ordered_rois.extend(present)

    # ── Compute plot values ──────────────────────────────────────────────────
    def get_vals(roi):
        nm, vm = roi_stats[roi]["mean"]
        ns, vs = roi_stats[roi]["std"]
        noun_dom = nm >= vm
        plot_mean = nm if noun_dom else 100 - vm   # mirror verbs left of 50%
        plot_std  = ns if noun_dom else vs
        color     = noun_color if noun_dom else verb_color
        return plot_mean, plot_std, color

    # ── Layout: assign a y position to each entry ───────────────────────────
    # Sentinels (None) get a small gap; group headers sit just above their block
    GAP = 0.5   # extra vertical space for dividers

    y_positions = {}   # roi_name -> float y
    y          = 0.0
    y_ticks    = []    # (y, label) for actual ROIs
    y_dividers = []    # y positions of divider lines
    y_headers  = []    # (y, label) for group header text

    for entry in ordered_rois:
        if entry is None:
            y_dividers.append(y - GAP/2)  # divider line goes in the middle of the gap
            y += GAP
        else:
            if len(ordered_rois) > 1:
                # Check if this roi immediately follows a header sentinel
                idx = ordered_rois.index(entry)  # first occurrence is fine
                if idx in group_headers or (idx > 0 and ordered_rois[idx - 1] is None and
                                             ordered_rois.index(entry) in
                                             {i + 1 for i in range(len(ordered_rois))
                                              if ordered_rois[i] is None}):
                    pass  # header handled below
            y_positions[entry] = y
            y_ticks.append((y, entry))
            y += 1.0

    # Re-derive header y positions (just above first ROI in each group)
    header_ys = {}
    for sentinel_idx, label in group_headers.items():
        # Find the first real ROI after this sentinel
        for j in range(sentinel_idx, len(ordered_rois)):
            if ordered_rois[j] is not None:
                header_ys[label] = y_positions[ordered_rois[j]]
                break

    total_height = y  # total y extent

    # ── Draw ─────────────────────────────────────────────────────────────────
    fig_height = max(5, total_height * 0.38 + 1.8)
    fig, ax = plt.subplots(figsize=(7, fig_height))

    # Shaded half-panels
    ax.axvspan(50, 100, color=noun_color, alpha=0.04, zorder=0)
    ax.axvspan(0,  50,  color=verb_color, alpha=0.04, zorder=0)

    # Chance line
    ax.axvline(50, color="gray", linewidth=0.8, linestyle="--", zorder=1)

    # Group header labels — inside the plot, left-aligned, just above the divider
    for label, yh in header_ys.items():
        ax.text(
            0.01, yh - GAP+ 0.3,       # small nudge above the first ROI in the group
            label,
            transform=ax.get_yaxis_transform(),
            fontsize=7.5, fontweight="bold",
            color="gray",
            ha="left", va="bottom",
        )

    # Group dividers — draw these after headers so they sit above the label
    for yd in y_dividers:
        ax.axhline(yd, color="lightgray", linewidth=0.8, linestyle="-", zorder=1)

    # Data points
    for roi, ypos in y_positions.items():
        mean, std, color = get_vals(roi)
        ax.errorbar(
            mean, ypos,
            xerr=std,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=3,
            capthick=1.2,
            markersize=5,
            zorder=2,
        )

    # Axes formatting
    ax.set_yticks([y for y, _ in y_ticks])
    ax.set_yticklabels([lbl for _, lbl in y_ticks], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.8, total_height - 0.2)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}%"))
    ax.set_xlabel("Winning Voxels (%)", fontsize=10)
    ax.set_title(
        f"ROI Dominant Category (mean ± SD) — {modality} (n={n_subjects})",
        fontsize=11, pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, linewidth=0.5, color="lightgray", zorder=0)
    ax.set_axisbelow(True)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=verb_color,
                   markersize=7, label="← Verb dominant"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=noun_color,
                   markersize=7, label="Noun dominant →"),
        plt.Line2D([0], [0], color="gray", linewidth=0.8, linestyle="--",
                   label="50% (chance)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    if output_dir:
        plt.savefig(output_dir / f"roi_dominance_{modality}.pdf")
    plt.show()
      
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quantitative analysis of flatmaps")
    parser.add_argument("--subjects", nargs="+", default=config.SUBJECTS, help="List of subjects")
    parser.add_argument("--modalities", nargs="+", default=config.MODALITIES, help="List of modalities")
    parser.add_argument("--rois", nargs="+", default=config.ROI, help="List of ROIs to analyze")
    parser.add_argument("--output_dir", default=config.OUTPUT_DIR/'figures', help="Directory to save output figures")  
    
    # Hybrid threshold arguments
    parser.add_argument("--min_voxels", type=int, default=10, help="Absolute floor for statistical stability")
    parser.add_argument("--min_roi_pct", type=float, default=0.10, help="Minimum proportion of the ROI that must be valid")
    
    args = parser.parse_args()
    
    for modality in args.modalities:
        roi_subject_values = {roi: [] for roi in args.rois}

        for subject in args.subjects:
            results = load_results(
                config.OUTPUT_DIR / "results_actual",
                modality,
                subject,
            )
            mapper = config.MAPPER_PATH / f"{subject}_mappers.hdf"

            verbs = get_significant_voxels(results, "verb")
            nouns = get_significant_voxels(results, "noun")

            for roi in args.rois:
                try:
                    mask = load_roi_mask(mapper, roi)
                except KeyError:
                    print(f"ROI {roi} not found for {subject}; skipping.")
                    continue

                percentages, n_valid, roi_size = percentage(nouns, verbs, mask, mapper)
                
                # Apply hybrid threshold
                valid_pct = n_valid / roi_size if roi_size > 0 else 0
                
                if n_valid < args.min_voxels or valid_pct < args.min_roi_pct:
                    percentages = np.array([np.nan, np.nan], dtype=float)
                    
                roi_subject_values[roi].append(percentages)

        roi_stats = {}
        for roi in args.rois:
            values = np.array(roi_subject_values[roi], dtype=float)
            if values.size == 0:
                print(f"No valid data collected for ROI {roi} in modality {modality}.")
                continue

            # Drop subject rows that are entirely NaN (no valid comparisons OR under threshold).
            values = values[~np.all(np.isnan(values), axis=1)]
            if values.size == 0:
                print(
                    f"Skipping ROI {roi} in modality {modality}: All subjects were under the threshold of {args.min_voxels} voxels."
                )
                continue

            roi_stats[roi] = {
                "mean": np.nanmean(values, axis=0),
                "std": np.nanstd(values, axis=0),
            }

        plot_percentage(roi_stats, modality, len(args.subjects), config.ROI_GROUPS_, output_dir=args.output_dir)