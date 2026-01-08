import numpy as np
import scipy.sparse
import h5py
from h5py._hl.dataset import Dataset
from h5py._hl.group import Group
import os
import matplotlib.pyplot as plt

def map_to_flat(voxels, mapper_file):
    """Generate flatmap image for an individual subject from voxel array

    This function maps a list of voxels into a flattened representation
    of an individual subject's brain.

    Parameters
    ----------
    voxels: array
        n x 1 array of voxel values to be mapped
    mapper_file: string
        file containing mapping arrays

    Returns
    -------
    image : array
        flatmap image, (n x 1024)

    By Mark Lescroart

    """
    pixmap = load_sparse_array(mapper_file, 'voxel_to_flatmap')
    with h5py.File(mapper_file, mode='r') as hf:
        pixmask = hf['flatmap_mask'][()]
    badmask = np.array(pixmap.sum(1) > 0).ravel()
    img = (np.nan * np.ones(pixmask.shape)).astype(voxels.dtype)
    mimg = (np.nan * np.ones(badmask.shape)).astype(voxels.dtype)
    mimg[badmask] = (pixmap * voxels.ravel())[badmask].astype(mimg.dtype)
    img[pixmask] = mimg
    return img.T[::-1]

def plot_correlation_on_flatmap(
    subject: str,
    modality: str,
    mode: str,
    r: np.ndarray,
    r_sig: np.ndarray,
    save_fig: bool = True,
    fdir: str = ".",
):
    """
    Plot correlation values on a flatmap representation for a given subject and modality.
    This function creates a two-panel visualization showing both significant and all correlation
    values mapped onto a flattened cortical surface representation. The correlation data is
    mapped from 3D to 2D using subject-specific mapper files.
    Args:
        subject (str): Subject identifier used to locate the mapper file and label outputs.
        modality (str): Modality identifier (e.g., imaging modality or data type) for labeling.
        mode (str): Analysis mode identifier used in the output filename.
        r (np.ndarray): Array of correlation coefficients to be visualized.
        r_sig (np.ndarray): Array of significant correlation coefficients (filtered by statistical threshold).
        save_fig (bool, optional): Whether to save the figure to disk. Defaults to True.
        fdir (str, optional): Directory path where mapper files are located. Defaults to ".".
    Returns:
        None
    Side Effects:
        - Creates 'outputs' directory if it doesn't exist
        - Saves numpy arrays of correlation values and flatmap to disk
        - Saves visualization as PNG file if save_fig is True
        - Displays the plot using plt.show()
    Note:
        - Uses a hot colormap with white for bad values and gray (#555555) for values below minimum
        - Both plots use vmin=0 and vmax=0.4 for consistent scaling
        - Output files are saved with naming pattern: outputs/subject{subject}_{modality}_*.npy/png
    """

    cmap_ = plt.cm.hot
    cmap_.set_bad(color="white")
    cmap_.set_under(color="#555555")

    map_file = os.path.join(fdir, "mappers", f"{subject}_mappers.hdf")
    flatmap_sig = map_to_flat(r_sig, map_file)
    flatmap = map_to_flat(r, map_file)

    os.makedirs("outputs", exist_ok=True)
    np.save(f"outputs/subject{subject}_{modality}_r.npy", r)
    np.save(f"outputs/subject{subject}_{modality}_flatmap.npy", flatmap)

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    axes[0].imshow(flatmap_sig, cmap=cmap_, vmin=0, vmax=0.4)
    axes[0].axis("off")
    axes[0].set_title(f"Significant r values: subject {subject}, {modality}")

    axes[1].imshow(flatmap, cmap=cmap_, vmin=0, vmax=0.4)
    axes[1].axis("off")
    axes[1].set_title(f"All r values: subject {subject}, {modality}")

    plt.tight_layout()
    if save_fig:
        output_path = f"outputs/images/{subject}_{modality}_{mode}.png"
        print(f"Saving flatmap figure to {output_path}")
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.show()