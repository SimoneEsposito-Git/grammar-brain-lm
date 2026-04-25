import numpy as np
import scipy.sparse
import h5py
from h5py._hl.dataset import Dataset
from h5py._hl.group import Group
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable
from functools import lru_cache

# ============================================================================
# region UTILITY FUNCTIONS
# ============================================================================

VC_ROIS = ["V1", "V2", "V3", "V3A", "V3B", "V4", "V7", "VO","FO", "FFA", "LO", "IPS"]
def smooth_flatmap(data, sigma=2.0):
    """Applies a NaN-safe Gaussian blur to a 2D flatmap."""
    v = data.copy()
    v[np.isnan(data)] = 0
    w = 1.0 - np.isnan(data).astype(float)
    vv = gaussian_filter(v, sigma=sigma)
    ww = gaussian_filter(w, sigma=sigma)
    smoothed = vv / (ww + 1e-10)
    smoothed[np.isnan(data)] = np.nan
    return smoothed

def get_bivariate_color(val1, val2, vmin=0, vmax=0.4):
    """Bivariate color mapping: val1 (Blue) vs val2 (Orange)"""
    x = np.clip((val1 - vmin) / (vmax - vmin), 0, 1)
    y = np.clip((val2 - vmin) / (vmax - vmin), 0, 1)
    rgb = np.zeros((*x.shape, 3))
    rgb[..., 0] = y 
    rgb[..., 1] = (x + y) * 0.4 + (x * y) * 0.2
    rgb[..., 2] = x
    rgb = np.power(rgb, 0.8)
    return np.clip(rgb, 0, 1)

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

def _overlay_flatmap_rois(rgb_map, map_file, roi_index=0):
    """Composite ROI boundaries directly into the RGB array.
    
    Args:
        rgb_map: RGBA array of shape (H, W, 4) — modified in place
        map_file: path to the HDF mapper file
        roi_index: which ROI layer to overlay
    """
    with h5py.File(map_file) as hf:
        overlay = hf["flatmap_rois"][:, :, roi_index]
    overlay = np.rot90(overlay, k=1)

    roi_mask = overlay > 0
    rgb_map[roi_mask, 0] = 0.8
    rgb_map[roi_mask, 1] = 0.8
    rgb_map[roi_mask, 2] = 0.8
    rgb_map[roi_mask, 3] = 1.0

# endregion
# ============================================================================
# region MAPPER: Encapsulates mapper file logic with caching
# ============================================================================

@dataclass
class FlatmapMapper:
    """Manages mapper file I/O and provides caching"""
    subject: str
    mapper_dir: str
    
    def __post_init__(self):
        self.map_file = os.path.join(self.mapper_dir, f"{self.subject}_mappers.hdf")
        if not os.path.exists(self.map_file):
            raise FileNotFoundError(f"Mapper file not found: {self.map_file}")
    
    @lru_cache(maxsize=1)
    def _load_pixmap(self):
        """Lazy load and cache sparse pixmap"""
        return load_sparse_array(self.map_file, "voxel_to_flatmap")
    
    @lru_cache(maxsize=1)
    def _load_mask(self):
        """Lazy load and cache flatmap mask"""
        with h5py.File(self.map_file, mode="r") as hf:
            return hf["flatmap_mask"][()]
    
    def to_flatmap(self, voxels: np.ndarray) -> np.ndarray:
        """Convert voxel data to flatmap space"""
        pixmap = load_sparse_array(self.map_file, "voxel_to_flatmap")
        with h5py.File(self.map_file, mode="r") as hf:
            pixmask = hf["flatmap_mask"][()]
        badmask = np.array(pixmap.sum(1) > 0).ravel()
        img = (np.nan * np.ones(pixmask.shape)).astype(voxels.dtype)
        mimg = (np.nan * np.ones(badmask.shape)).astype(voxels.dtype)
        mimg[badmask] = (pixmap * voxels.ravel())[badmask].astype(mimg.dtype)
        img[pixmask] = mimg
        return img.T[::-1]
    
    def get_brain_mask(self) -> np.ndarray:
        """Get binary brain mask"""
        with h5py.File(self.map_file, mode="r") as hf:
            return hf["flatmap_mask"][()].T[::-1]
    
    def get_brain_bkg(self, color: Tuple[float, float, float]) -> np.ndarray:
        """Get flatmap curvature data"""
        if color is not None:
            return np.tile(color, (*self.get_brain_mask().shape, 1))
        with h5py.File(self.map_file, mode="r") as hf:
            try:
                curvature = hf["flatmap_curvature"][()].T[::-1]
            except:
                curvature = np.zeros(self.get_brain_mask().shape)
            return np.stack([curvature, curvature, curvature], axis=-1)
    
    def mask_vc(self, voxels: np.ndarray) -> np.ndarray:
        """Get visual cortex mask"""
        masked = voxels.copy()
        with h5py.File(self.map_file, mode="r") as hf:
            for roi in VC_ROIS:
                name = "roi_mask_%s" % roi
                if name in hf:
                    roi_mask = hf[name][()].astype(bool)
                    
                    masked[roi_mask] = np.nan
        return masked

# endregion
# ============================================================================
# region COLOR STRATEGIES: Pluggable color generation
# ============================================================================

class ColorStrategy(ABC):
    """Abstract base for color generation strategies
    
    Strategies are callables that take flatmaps and return RGB arrays.
    """
    
    @abstractmethod
    def __call__(self, flatmaps: List[np.ndarray], background: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Generate RGB array. 
        
        Returns:
            shape (H, W, 3) array in [0, 1] range
        """
        pass
    
    def get_legend_type(self) -> Optional[str]:
        """Return legend type: 'colorbar', 'categorical', 'bivariate', or None"""
        return None
    
    def get_legend_data(self) -> Optional[Dict]:
        """Return metadata for legend rendering"""
        return None


class SingleCorrelationStrategy(ColorStrategy):
    """Single correlation map with hot colormap"""
    
    def __init__(self, vmin: float = 0, vmax: float = 0.5):
        self.vmin = vmin
        self.vmax = vmax
    
    def __call__(self, flatmaps: List[np.ndarray], background: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        flatmap = flatmaps[0]
        cmap = plt.cm.hot
        norm = plt.Normalize(vmin=self.vmin, vmax=self.vmax, clip=True)
        rgb = cmap(norm(flatmap))[..., :3]
        if background is not None:
            rgb[np.isnan(flatmap)] = background[np.isnan(flatmap)]
        return rgb
    
    def get_legend_type(self) -> str:
        return 'colorbar'
    
    def get_legend_data(self) -> Dict:
        return {
            'vmin': self.vmin,
            'vmax': self.vmax,
            'cmap': 'hot',
            'label': 'Correlation'
        }

class ContrastStrategy(ColorStrategy):
    """Contrast between two maps: (corr_1 - corr_2) with diverging colormap"""
    
    def __init__(self, vmin: float = -0.5, vmax: float = 0.5):
        self.vmin = vmin
        self.vmax = vmax
        self.cmap = plt.cm.coolwarm
    
    def __call__(self, flatmaps: List[np.ndarray], background: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        contrast = flatmaps[0] - flatmaps[1]
        cmap = self.cmap
        norm = plt.Normalize(vmin=self.vmin, vmax=self.vmax, clip=True)
        rgb = cmap(norm(contrast))[..., :3]
        if background is not None:
            rgb[np.isnan(contrast)] = background[np.isnan(contrast)]*0.7
        return rgb
    
    def get_legend_type(self) -> str:
        return 'colorbar'
    
    def get_legend_data(self) -> Dict:
        return {
            'vmin': self.vmin,
            'vmax': self.vmax,
            'cmap': self.cmap,
            'label': 'Contrast'
        }

class BivariateStrategy(ColorStrategy):
    """Bivariate color mapping: corr_1 (Blue) vs corr_2 (Orange)"""
    
    def __init__(self, vmax: float = 0.5, labels: Tuple[str, str] = ("Model 1", "Model 2")):
        self.vmax = vmax
        self.labels = labels    
    
    def __call__(self, flatmaps: List[np.ndarray], background: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        rgb_map = get_bivariate_color(flatmaps[0], flatmaps[1], vmin=0, vmax=self.vmax)
        mask = np.isnan(flatmaps[0]) | np.isnan(flatmaps[1])
        rgb_map[mask] = background[mask]*0.7 if background is not None else [1, 1, 1]
        return rgb_map
    
    def get_legend_type(self) -> str:
        return 'bivariate'
    
    def get_legend_data(self) -> Dict:
        return {
            'vmax': self.vmax,
            'labels': self.labels
        }


class LoserTakesItAllStrategy(ColorStrategy):
    """Winner (best) is shown, colored by model"""
    
    def __init__(self, names: List[str], palette: Optional[List[str]] = None, rank: int = 0):
        self.names = names
        self.rank = rank
        if palette is None:
            palette = ["#007ABF", "#FF7A00", "#3CB44B", "#E6194B", "#FFE119", "#911EB4"]
            #palette = ["#015FDA","#F6931D", "#03FF24", "#FF102F", "#E1DF01", "#8D09FE"]
        self.colors = [mcolors.to_rgb(c) for c in palette[:len(names)]]
    
    def __call__(self, flatmaps: List[np.ndarray], background: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        baseline = flatmaps[-1]
        stacked = np.stack(flatmaps[:-1], axis=0)-baseline
        with np.errstate(invalid='ignore'):
            winner_indices = np.argsort(stacked, axis=0)[self.rank]
    
        h, w = flatmaps[0].shape
        rgb_map = np.zeros((h, w, 3))
        brain_mask = ~np.all(np.isnan(stacked), axis=0)
        
        for i, rgb in enumerate(self.colors):
            mask = (winner_indices == i) & brain_mask
            strength = flatmaps[i][mask]/np.nanpercentile(flatmaps[i][mask], 95)
            strength = np.nan_to_num(np.clip(strength, 0, 1))
            rgb_map[mask] = np.array(rgb) * strength[:, np.newaxis] + (1 - strength[:, np.newaxis]) * (background[mask] if background is not None else [1, 1, 1])
        
        rgb_map[~brain_mask] = background[~brain_mask]*0.7 if background is not None else [1, 1, 1]
        return rgb_map
    
    def get_legend_type(self) -> str:
        return 'categorical'
    
    def get_legend_data(self) -> Dict:
        # Create Line2D objects for legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor=mcolors.to_hex(c),
                   markersize=8, label=name)
            for name, c in zip(self.names, self.colors)
        ]
        return {'legend_elements': legend_elements}

# endregion
# ============================================================================
# region FIGURE BUILDERS: Layout and rendering
# ============================================================================

@dataclass
class FigureConfig:
    """Configuration for figure rendering"""
    figsize: Tuple[int, int] = (12, 9)
    dpi: int = 150
    mask_vc: bool = True
    smooth: bool = False
    sigma: float = 2.0
    show_regions: bool = False
    show: bool = True
    save: bool = True
    output_dir: str = "."
    # Fixed layout parameters for consistent sizing
    title_height: float = 0.08  # 8% of figure for title
    legend_height: float = 0.08  # 8% of figure for legend
    legend_padding: float = 0.0  # padding inside legend band
    image_left: float = 0.05
    image_right: float = 0.95
    image_bottom_padding: float = 0.0  # Small padding above legend


class FigureBuilder(ABC):
    """Abstract base for figure composition"""
    
    @abstractmethod
    def build(self) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Create figure with appropriate layout"""
        pass
    
    @abstractmethod
    def render(
        self,
        fig: plt.Figure,
        axes: List[plt.Axes],
        rgb_map: np.ndarray,
        title: str = "",
        subject: str = "",
        **kwargs,
    ):
        """Render the rgb_map onto axes"""
        pass
    
    @abstractmethod
    def finalize(
        self,
        fig: plt.Figure,
        axes: List[plt.Axes],
        output_filename: str = "",
        **kwargs,
    ):
        """Save/show the figure"""
        pass


class SinglePanelBuilder(FigureBuilder):
    """Single panel figure with optional colorbar"""
    
    def __init__(self, config: FigureConfig, mapper: Optional[FlatmapMapper] = None):
        self.config = config
        self.mapper = mapper
    
    def build(self) -> Tuple[plt.Figure, List[plt.Axes]]:
        fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Calculate image area position
        image_bottom = self.config.legend_height + self.config.image_bottom_padding
        image_top = 1.0 - self.config.title_height
        image_height = image_top - image_bottom
        image_width = self.config.image_right - self.config.image_left
        
        # Create main axis with fixed position and size
        ax = fig.add_axes([
            self.config.image_left,
            image_bottom,
            image_width,
            image_height
        ])
        
        return fig, [ax]
    
    def render(
        self,
        fig: plt.Figure,
        axes: List[plt.Axes],
        rgb_map: np.ndarray,
        title: str = "",
        subject: str = "",
        colorbar_label: str = "",
        colorbar_data: Optional[Tuple] = None,
        strategy: Optional['ColorStrategy'] = None,
        **kwargs,
    ):
        ax = axes[0]
        ax.imshow(rgb_map, interpolation='none')
        ax.set_facecolor('none')
        fig.patch.set_alpha(0)
        ax.axis("off")
        
        # Use suptitle for consistent positioning
        if title:
            fig.suptitle(title, fontsize=12, y=0.95)
        if subject:
            fig.text(0.5, 0.75, subject, ha='center', fontsize=12)
        
        # Auto-generate colorbar for single correlation in fixed legend area
        if strategy is not None and strategy.get_legend_type() == 'colorbar':
            legend_data = strategy.get_legend_data()
            vmin = legend_data.get('vmin', 0)
            vmax = legend_data.get('vmax', 0.5)
            cmap_name = legend_data.get('cmap', 'hot')
            label = legend_data.get('label', 'Value')
            
            cmap = plt.get_cmap(cmap_name)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            
            # Create colorbar axis centered in the legend band
            legend_inner_bottom = self.config.legend_padding
            legend_inner_top = self.config.legend_height - self.config.legend_padding
            legend_inner_height = legend_inner_top - legend_inner_bottom
            cbar_height = min(0.03, legend_inner_height * 0.4)
            cbar_y = legend_inner_bottom + (legend_inner_height - cbar_height) / 2
            cbar_ax = fig.add_axes([0.25, cbar_y, 0.5, cbar_height])
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label(label, fontsize=8)
        
        # Fallback for explicit colorbar_data
        elif colorbar_data is not None:
            flatmap, norm, cmap = colorbar_data
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            legend_inner_bottom = self.config.legend_padding
            legend_inner_top = self.config.legend_height - self.config.legend_padding
            legend_inner_height = legend_inner_top - legend_inner_bottom
            cbar_height = min(0.03, legend_inner_height * 0.4)
            cbar_y = legend_inner_bottom + (legend_inner_height - cbar_height) / 2
            cbar_ax = fig.add_axes([0.25, cbar_y, 0.5, cbar_height])
            cbar = fig.colorbar(sm, cax=cbar_ax, label=colorbar_label, orientation='horizontal')
    
    def finalize(
        self,
        fig: plt.Figure,
        axes: List[plt.Axes],
        output_filename: str = "",
        legend_elements: Optional[List] = None,
        strategy: Optional['ColorStrategy'] = None,
        **kwargs,
    ):
        # Auto-generate categorical legend from strategy
        if strategy is not None and strategy.get_legend_type() == 'categorical':
            legend_data = strategy.get_legend_data()
            legend_elements = legend_data.get('legend_elements')
        
        if legend_elements and kwargs.get('show_legend', True):
            print("Adding legend with elements:", legend_elements)
            # Position legend centered in the padded legend band
            legend_inner_bottom = self.config.legend_padding
            legend_inner_top = self.config.legend_height - self.config.legend_padding
            legend_center_y = (legend_inner_bottom + legend_inner_top) / 2
            fig.legend(
                handles=legend_elements,
                loc='center',
                ncol=min(len(legend_elements), 3),
                bbox_to_anchor=(0.5, legend_center_y),
                fontsize=6,
                handletextpad=0,
                columnspacing=0.1,
                frameon=False
            )
        # Don't use tight_layout since we're using manual positioning
        
        if self.config.save:
            os.makedirs(self.config.output_dir, exist_ok=True)
            if not output_filename:
                output_filename = "flatmap.pdf"
            output_path = os.path.join(self.config.output_dir, output_filename)
            plt.savefig(output_path, format='pdf', bbox_inches='tight', transparent=True)
            print(f"Saved to {output_path}")
        
        if self.config.show:
            plt.show()
        else:
            plt.close()


class BivariateLegendBuilder(SinglePanelBuilder):
    """Single panel with 2D bivariate legend"""
    
    def render(
        self,
        fig: plt.Figure,
        axes: List[plt.Axes],
        rgb_map: np.ndarray,
        title: str = "",
        subject: str = "",
        legend_vmax: float = 0.5,
        legend_labels: Tuple[str, str] = ("X", "Y"),
        strategy: Optional['ColorStrategy'] = None,
        **kwargs,
    ):
        # Auto-extract legend_vmax from strategy if available
        if strategy is not None and strategy.get_legend_type() == 'bivariate':
            legend_data = strategy.get_legend_data()
            legend_vmax = legend_data.get('vmax', 0.5)
            legend_labels = tuple(legend_data.get('labels', ["X", "Y"]))
        
        # Call parent render but skip strategy processing (we handle it manually)
        ax = axes[0]
        ax.imshow(rgb_map, interpolation='none')
        ax.axis("off")
        
        if title:
            fig.suptitle(title, fontsize=12, y=0.95)
        if subject:
            fig.text(0.5, 0.75, subject, ha='center', fontsize=12)
            
        if not kwargs.get('show_legend', True):
            return 
        # Add 2D legend inside the fixed legend area to avoid clipping
        legend_inner_bottom = self.config.legend_padding
        legend_inner_top = self.config.legend_height - self.config.legend_padding
        legend_inner_height = legend_inner_top - legend_inner_bottom
        legend_size = 0.2
        legend_x = 0.85 - legend_size / 2
        legend_y = 0.05
        
        ax_inset = fig.add_axes([legend_x, legend_y, legend_size, legend_size])
        x_grid, y_grid = np.meshgrid(
            np.linspace(0, legend_vmax, 50),
            np.linspace(0, legend_vmax, 50)
        )
        legend_rgb = get_bivariate_color(x_grid, y_grid, vmin=0, vmax=legend_vmax)
        ax_inset.imshow(legend_rgb, origin='lower', extent=[0, legend_vmax, 0, legend_vmax])
        ax_inset.set_xticks([0, legend_vmax])
        ax_inset.set_xticklabels(['0', str(legend_vmax)], ha='left')  # left-align so '0' hugs left edge

        ax_inset.set_yticks([0, legend_vmax])
        ax_inset.set_yticklabels(['0', str(legend_vmax)], va='bottom')  # bottom-align so '0' hugs bottom edge
        for label, ha in zip(ax_inset.get_xticklabels(), ['left', 'right']):
            label.set_ha(ha)

        for label, va in zip(ax_inset.get_yticklabels(), ['bottom', 'top']):
            label.set_va(va)
        ax_inset.tick_params(labelsize=4, length=0, pad=2) 
        ax_inset.spines['top'].set_visible(False)
        ax_inset.spines['right'].set_visible(False)
        ax_inset.spines['bottom'].set_visible(False)
        ax_inset.spines['left'].set_visible(False)
        # Place labels inside the legend band to prevent clipping
        fig.text(
            legend_x + legend_size / 2,
            legend_y - 0.1,
            legend_labels[0],
            ha="center",
            va="bottom",
            fontsize=6,
        )
        fig.text(
            legend_x,
            legend_y + legend_size / 2,
            legend_labels[1],
            ha="right",
            va="center",
            rotation=90,
            fontsize=6,
        )

# endregion
# ============================================================================
# region UNIFIED PLOTTER: One class for all visualizations
# ============================================================================

@dataclass
class FlatmapPlotter:
    """
    Universal flatmap plotter using dependency injection.
    
    No subclassing needed—just provide your strategy and builder.
    """
    mapper: FlatmapMapper
    config: FigureConfig = field(default_factory=FigureConfig)
    builder: Optional[FigureBuilder] = None
    
    def plot(
        self,
        data: List[np.ndarray],
        strategy: ColorStrategy,
        title: str = "",
        subject: str = "",
        output_filename: str = "",
        builder_kwargs: Optional[Dict] = None,
        background_color: Optional[Tuple[float, float, float]] = None,
        **kwargs,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot data using the provided color strategy and figure builder.
        
        Parameters
        ----------
        data : List[np.ndarray]
            List of voxel arrays to map and visualize
        strategy : ColorStrategy
            Color generation strategy
        title : str
            Plot title
        subject : str
            Subject identifier
        output_filename : str
            Output filename (if save=True)
        builder_kwargs : dict
            Additional kwargs passed to the builder's render/finalize
        
        Returns
        -------
        fig, rgb_map
            Figure and RGB array
        """
        builder_kwargs = builder_kwargs or {}
        
        # Map to flatmaps
        if self.config.mask_vc:
            data = [self.mapper.mask_vc(d) for d in data]
            
        flatmaps = [self.mapper.to_flatmap(d) for d in data]
        
        # Smooth if requested
        if self.config.smooth:
            flatmaps = [smooth_flatmap(fm, sigma=self.config.sigma) for fm in flatmaps]
        
        # Generate colors
        print(flatmaps[0])
        # Generate colors
        background = self.mapper.get_brain_bkg(background_color)
        rgb_map = strategy(flatmaps, background=background, **kwargs)
        alpha = self.mapper.get_brain_mask().astype(float)[..., np.newaxis]
        rgb_map = np.concatenate([rgb_map, alpha], axis=-1)

        # Composite ROIs into the array before rendering
        if self.config.show_regions:
            _overlay_flatmap_rois(rgb_map, self.mapper.map_file)
        # Auto-select builder based on strategy type
        if self.builder is None:
            strategy_type = strategy.get_legend_type()
            if strategy_type == 'bivariate':
                builder = BivariateLegendBuilder(self.config, self.mapper)
            else:
                builder = SinglePanelBuilder(self.config, self.mapper)
        else:
            builder = self.builder
        
        fig, axes = builder.build()
        
        # Render with strategy passed for auto-legend generation
        builder.render(fig, axes, rgb_map, title=title, subject=subject, strategy=strategy, **builder_kwargs)
        
        # Finalize with strategy passed for auto-legend generation
        builder.finalize(fig, axes, output_filename=output_filename, strategy=strategy, **builder_kwargs)
        
        return fig, rgb_map

# endregion
# ============================================================================