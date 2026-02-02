import numpy as np
import scipy.sparse
import h5py
from h5py._hl.dataset import Dataset
from h5py._hl.group import Group
import os
import matplotlib.pyplot as plt


def load_sparse_array(fname, varname):
    """Load a numpy sparse array from an hdf file

    Parameters
    ----------
    fname: string
        file name containing array to be loaded
    varname: string
        name of variable to be loaded

    Notes
    -----
    This function relies on variables being stored with specific naming
    conventions, so cannot be used to load arbitrary sparse arrays.

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
    pixmap = load_sparse_array(mapper_file, "voxel_to_flatmap")
    with h5py.File(mapper_file, mode="r") as hf:
        pixmask = hf["flatmap_mask"][()]
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
    results: dict,
    mapper_path: str,
    output_dir: str = "outputs/images",
    show_plot: str = True,
    show_nonsig: bool = True,
    save_fig: bool = True,
    contrast=False,
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
    """
    if contrast:
        cmap_ = plt.cm.berlin
    else:
        cmap_ = plt.cm.hot
        cmap_.set_bad(color="white")
        cmap_.set_under(color="#555555")

    r = results["r"]
    r_sig = r.copy()
    r_sig[results['fdr']['r']['excluded_voxels_indices']] = 0
    
    map_file = os.path.join(mapper_path, f"{subject}_mappers.hdf")
    flatmap_sig = map_to_flat(r_sig, map_file)
    flatmap = map_to_flat(r, map_file)

    os.makedirs("outputs", exist_ok=True)

    if show_nonsig:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]

    if contrast:
        vmin_ = -0.2
        vmax_ = 0.2
    else:
        vmin_ = 0
        vmin_ns = -0.5
        vmax_ = 0.5

    axes[0].imshow(flatmap_sig, cmap=cmap_, vmin=vmin_, vmax=vmax_)
    axes[0].axis("off")
    axes[0].set_title(f"Significant r values: subject {subject}, {modality}. (FDR corrected, permutation test)")
    
    if contrast:
        im = axes[0].imshow(flatmap_sig, cmap=cmap_, vmin=vmin_, vmax=vmax_)
        cbar = plt.colorbar(im, ax=axes[0], label='Correlation Difference(r1 - r2)', orientation='horizontal', pad=0.05, shrink=0.6)
        cbar.set_ticks([vmin_, vmax_])
    else:
        im = axes[0].imshow(flatmap_sig, cmap=cmap_, vmin=vmin_, vmax=vmax_)
        cbar = plt.colorbar(im, ax=axes[0], label='Correlation Coefficient (r)', orientation='horizontal', pad=0.05, shrink=0.6)
        cbar.set_ticks([vmin_, vmax_])
    
    if show_nonsig:
        axes[1].imshow(flatmap, cmap=cmap_, vmin=vmin_, vmax=vmax_)
        axes[1].axis("off")
        axes[1].set_title(f"All r values: subject {subject}, {modality}")
        im = axes[1].imshow(flatmap, cmap=cmap_, vmin=vmin_, vmax=vmax_)
        cbar = plt.colorbar(im, ax=axes[1], label='Correlation Coefficient (r)', orientation='horizontal', pad=0.05, shrink=0.6)
        cbar.set_ticks([vmin_, vmax_])

    plt.tight_layout()
    if save_fig:
        output_path = os.path.join(output_dir, f"{subject}_{modality}_{mode}.png")
        print(f"Saving flatmap figure to {output_path}")
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        
    if show_plot:
        plt.show()
