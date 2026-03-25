import numpy as np
import h5py
import matplotlib.pyplot as plt
from data_loading.file_io import load_sparse_array, load_results
from data_loading import config

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
        mask = hf[f"roi_mask_{roi}"][()]
    return mask.T[::-1]

def percentage(flat_a: np.ndarray, flat_b: np.ndarray, mask: np.ndarray) -> float:
    flat_a = flat_a[mask]
    flat_b = flat_b[mask]

    stacked = np.stack([flat_a, flat_b], axis=0)
    with np.errstate(invalid='ignore'):
        winner_indices = np.argsort(stacked, axis=0)[0]
    # Count how many times each flatmap wins
    counts = np.bincount(winner_indices.flatten(), minlength=stacked.shape[0])
    # Calculate percentage for each flatmap
    percentages = counts / winner_indices.size * 100
    return percentages

def plot_percentage(percentages: np.ndarray, labels: List[str]):
    plt.figure(figsize=(10, 6))
    plt.bar(labels, percentages)
    plt.ylabel("Percentage of Winning Voxels (%)")
    plt.title("Percentage of Voxels Where Each Flatmap Wins")
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quantitative analysis of flatmaps")
    parser.add_argument("--subjects", nargs="+", default=config.SUBJECTS, help="List of subjects")
    parser.add_argument("--modalities", nargs="+", default=config.MODALITIES, help="List of modalities")
    args = parser.parse_args()
    
    for subject in args.subjects:
        for modality in args.modalities:
            results = file_io.load_results(
                config.OUTPUT_DIR / "results_actual",
                args.modality,
                args.subject
            )
            verbs = get_significant_voxels(results, "verb")
            nouns = get_significant_voxels(results, "noun")
            verb_flat = map_to_flat(verbs, config.VOXEL_TO_FLATMAP_FILE)
            noun_flat = map_to_flat(nouns, config.VOXEL_TO_FLATMAP_FILE)
            percentages = percentage(verb_flat, noun_flat)
            plot_percentage(percentages, labels=["Verb Flatmap", "Noun Flatmap"])
    
    