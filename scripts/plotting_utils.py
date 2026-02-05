import numpy as np
import scipy.sparse
import h5py
from h5py._hl.dataset import Dataset
from h5py._hl.group import Group
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

def get_bivariate_color(val1, val2, vmin=0, vmax=0.4):
    """
    val1: Cross-modality (mapped to Blue)
    val2: Within-modality (mapped to Orange)
    """
    # Normalize to [0, 1]
    x = np.clip((val1 - vmin) / (vmax - vmin), 0, 1)
    y = np.clip((val2 - vmin) / (vmax - vmin), 0, 1)
    
    # Initialize RGB array
    rgb = np.zeros((*x.shape, 3))
    
    # Orange Component (High val2): Red=1.0, Green=0.5, Blue=0
    # Blue Component (High val1): Red=0, Green=0.5, Blue=1.0
    
    # Red channel: Driven by Within-modality (Orange)
    rgb[..., 0] = y 
    # Green channel: Provides the "brightness" and creates white when both are high
    rgb[..., 1] = (x + y) * 0.4 + (x * y) * 0.2
    # Blue channel: Driven by Cross-modality (Blue)
    rgb[..., 2] = x
    
    # Boost contrast: apply a slight power law to make colors 'pop'
    rgb = np.power(rgb, 0.8) 
    
    return np.clip(rgb, 0, 1)

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


def _overlay_flatmap_rois(ax, map_file, roi_index=0):
    """Overlay flatmap ROIs on a given axis"""
    with h5py.File(map_file) as hf:
        overlay = hf["flatmap_rois"][:, :, roi_index]
    overlay = np.rot90(overlay, k=1)  # counterclockwise rotation
    ax.imshow(
        overlay,
        cmap=mcolors.ListedColormap(["white"]),
        alpha=np.where(overlay > 0, 1.0, 0.0),
        interpolation="nearest",
    )

def plot_correlation_on_flatmap(
    subject: str,
    modality: str,
    mode: str,
    correlation: dict,
    mapper_dir: str,
    output_dir: str,
    title: str = "correlation flatmap",
    show: bool = True,
    save: bool = True,
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
    cmap_ = plt.cm.hot
    cmap_.set_bad(color="white")
    cmap_.set_under(color="#555555")
    
    map_file = os.path.join(mapper_dir, f"{subject}_mappers.hdf")
    flatmap = map_to_flat(correlation, map_file)

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    axes = [axes]

    vmin_ = 0
    vmax_ = 0.5

    axes[0].imshow(flatmap, cmap=cmap_, vmin=vmin_, vmax=vmax_)
    axes[0].axis("off")
    axes[0].set_title(title)
    
   
    im = axes[0].imshow(flatmap, cmap=cmap_, vmin=vmin_, vmax=vmax_)
    cbar = plt.colorbar(im, ax=axes[0], label='Correlation Coefficient (r)', orientation='horizontal', pad=0.05, shrink=0.6)
    cbar.set_ticks([vmin_, vmax_])

    _overlay_flatmap_rois(axes[0], map_file)
    plt.tight_layout()
    
    if save:
        output_path = os.path.join(output_dir, f"{subject}_{modality}_{mode}.png")
        print(f"Saving flatmap figure to {output_path}")
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        
    if show:
        plt.show()

def plot_bivariate_flatmap(
    subject: str,
    modality: str,
    corr_1: dict, # Model 1 (X)
    corr_2: dict, # Model 2 (Y)
    mode_1: str,
    mode_2: str,
    mapper_dir: str,
    output_dir: str,
    title: str = "Bivariate Correlation Map",
    show: bool = True,
    save: bool = True,
):
    map_file = os.path.join(mapper_dir, f"{subject}_mappers.hdf")
    
    # Map both sets of correlations to 2D space
    flat_cross = map_to_flat(corr_1, map_file)
    flat_within = map_to_flat(corr_2, map_file)

    # Create the RGB map
    # Assuming 'map_to_flat' returns NaNs or 0s for non-brain areas
    mask = np.isnan(flat_cross) | np.isnan(flat_within)
    
    # Generate RGB values using the bivariate logic
    rgb_map = get_bivariate_color(flat_cross, flat_within, vmin=0, vmax=0.4)
    
    # Set non-brain areas to white (or your preferred background)
    rgb_map[mask] = [1, 1, 1] 

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb_map)
    ax.axis("off")
    ax.set_title(title)

    _overlay_flatmap_rois(ax, map_file)
    # --- ADDING THE 2D LEGEND ---
    # This creates the small square key seen in your reference image
    ax_inset = fig.add_axes([0.7, 0.1, 0.15, 0.15]) # [left, bottom, width, height]
    x_grid, y_grid = np.meshgrid(np.linspace(0, 0.4, 50), np.linspace(0, 0.4, 50))
    legend_rgb = get_bivariate_color(x_grid, y_grid, vmin=0, vmax=0.4)
    ax_inset.imshow(legend_rgb, origin='lower', extent=[0, 0.4, 0, 0.4])
    ax_inset.set_xlabel(mode_1, fontsize=8)
    ax_inset.set_ylabel(mode_2, fontsize=8)
    ax_inset.set_xticks([0, 0.4])
    ax_inset.set_yticks([0, 0.4])

    if save:
        plt.savefig(os.path.join(output_dir, f"{subject}_{modality}_{mode_1}_vs_{mode_2}.png"), bbox_inches="tight")
    if show:
        plt.show()
        
def plot_loser_takes_it_all(
    subject: str,
    modality: str,
    correlations: list,  # list of arrays
    names: list,         # list of strings (e.g., ['Model A', 'Model B'])
    mapper_dir: str,
    output_dir: str,
    title: str = "Loser Takes It All Map",
    mask = None,
    regions: bool = False,
    show: bool = True,
    save: bool = True,
):
    # 1. Define high-contrast colors for the modes
    # Using a vibrant palette: Orange, Azure Blue, Bright Green, Purple
    palette = ["#FF8C00", "#00CCFF", "#33FF33", "#FF00FF"]
    colors = palette[:len(names)]
    
    map_file = os.path.join(mapper_dir, f"{subject}_mappers.hdf")
    
    # 2. Map all correlation arrays to the flatmap space
    flatmaps = []
    for corr in correlations:
        flatmaps.append(map_to_flat(corr, map_file))
    
    stacked = np.stack(flatmaps, axis=0)
    
    # 3. Logic: Find the index of the SECOND MINIMUM value for each voxel
    with np.errstate(invalid='ignore'):
        loser_indices = np.argsort(stacked, axis=0)[1]  # Second-lowest index
    
    # 4. Create the RGB Image
    h, w = flatmaps[0].shape
    rgb_map = np.zeros((h, w, 3))
    
    # Mask for valid brain data (where at least one model has a value)
    brain_mask = ~np.all(np.isnan(stacked), axis=0)
    background_mask = map_to_flat(np.ones_like(correlations[0]), map_file) == 1 
    
    rgb_map[background_mask] = [0, 0, 0]
    for i, hex_color in enumerate(colors):
        rgb = mcolors.to_rgb(hex_color)
        # Apply color where this index is the loser AND it's part of the brain
        mask = (loser_indices == i) & brain_mask
        # Scale color by correlation (0 = black, 0.4 = full color)
        correlation_strength = np.nan_to_num(np.clip(flatmaps[i][mask] / 0.4, 0, 1))
        rgb_map[mask] = np.array(rgb) * correlation_strength[:, np.newaxis]

    # Set background (non-brain) to a very dark grey for contrast
    # rgb_map[background_mask] = [0, 0, 0, 1]  # Opaque black background
    rgb_map[~brain_mask] = [1,1,1]
    rgb_map[background_mask & ~brain_mask] = [0, 0, 0]
    
    if mask is not None:
        flat_mask = map_to_flat(mask, map_file)
        rgb_map[~flat_mask.astype(bool)] = [0,0,0]  # Set masked-out areas to white

    # 5. Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(rgb_map, interpolation='none')
    ax.axis("off")
    ax.set_title(f"{title}\nSubject: {subject} | Modality: {modality}", 
                 color='white', fontsize=14, pad=20)

    if regions:
        _overlay_flatmap_rois(ax, map_file)
    # 6. Create Categorical Legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='none', label=f'{names[i]}',
               markerfacecolor=colors[i], markersize=12)
        for i in range(len(names))
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=len(names),
              bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        out_name = f"{subject}_{modality}_loser_map.png"
        plt.savefig(os.path.join(output_dir, out_name), bbox_inches="tight", dpi=200)
        
    if show:
        plt.show()
    else:
        plt.close()