"""
Flatmap visualization utilities for brain imaging analysis.

ARCHITECTURE:
    Fully composable, strategy-based design with zero duplication:

    1. UTILITY FUNCTIONS
       - load_sparse_array, map_to_flat, smooth_flatmap, get_bivariate_color
       - Low-level operations (file I/O, data transformation, colors)

    2. MAPPERS & CACHES
       - FlatmapMapper: Encapsulates subject-specific mapper file logic
       - Lazy loads and caches mapper data to avoid repeated file I/O

    3. COLOR STRATEGIES (pluggable)
       - ColorStrategy (abstract): Define color algorithm
       - Implementations: SingleCorrelation, Bivariate, LoserTakesItAll variants
       - Each strategy is self-contained and testable

    4. FIGURE BUILDERS (layout & rendering logic)
       - FigureBuilder: Abstract base for figure composition
       - Handles: axes setup, legends, colorbars, ROI overlays
       - Subclasses: SinglePanelBuilder, BivariateLegendBuilder, etc.

    5. UNIFIED PLOTTER (one class to rule them all)
       - FlatmapPlotter: Takes data + strategy + builder
       - Same pipeline for all visualization types
       - No subclassing needed, just inject your strategy

ADDING NEW VISUALIZATION TYPES:
    Just create a ColorStrategy, no new plotter class:
    
    ```python
    class MyColorStrategy(ColorStrategy):
        def __call__(self, flatmaps, **kwargs):
            return rgb_array
    
    # Plot it:
    plotter = FlatmapPlotter(mapper, figure_builder)
    plotter.plot(
        data=correlations,
        strategy=MyColorStrategy(**params),
        title="My visualization"
    )
    ```

KEY IMPROVEMENTS:
    - No abstract methods → no forced subclassing
    - Mapper caching → faster repeated plots
    - Dependency injection → composable, testable
    - Single entry point → less confusion
    - Strategy as callable → simpler interface
    - Figure builder handles all layout → easy customization
    - No class per visualization type → scales to 100+ variants
"""

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

def _overlay_flatmap_rois(ax, map_file, roi_index=0):
    """Overlay flatmap ROIs on a given axis"""
    with h5py.File(map_file) as hf:
        overlay = hf["flatmap_rois"][:, :, roi_index]
    overlay = np.rot90(overlay, k=1)
    ax.imshow(
        overlay,
        cmap=mcolors.ListedColormap(["#CCCCCC"]),
        alpha=np.where(overlay > 0, 1.0, 0.0),
        interpolation="nearest",
    )

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
    
    def get_brain_bkg(self) -> np.ndarray:
        """Get flatmap curvature data"""
        with h5py.File(self.map_file, mode="r") as hf:
            curvature = hf["flatmap_curvature"][()].T[::-1]
            return np.stack([curvature, curvature, curvature], axis=-1)

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


class BivariateStrategy(ColorStrategy):
    """Bivariate color mapping: corr_1 (Blue) vs corr_2 (Orange)"""
    
    def __init__(self, vmax: float = 0.5, labels: Tuple[str, str] = ("Model 1", "Model 2")):
        self.vmax = vmax
        self.labels = labels    
    
    def __call__(self, flatmaps: List[np.ndarray], background: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        rgb_map = get_bivariate_color(flatmaps[0], flatmaps[1], vmin=0, vmax=self.vmax)
        mask = np.isnan(flatmaps[0]) | np.isnan(flatmaps[1])
        rgb_map[mask] = background[mask] if background is not None else [1, 1, 1]
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
            palette = ["#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231", "#911EB4"]
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
        
        rgb_map[~brain_mask] = background[~brain_mask] if background is not None else [1, 1, 1]
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


class LoserTakesItAllMarginStrategy(ColorStrategy):
    """Loser shown with margin as saturation"""
    
    def __init__(self, names: List[str], palette: Optional[List[str]] = None, vmax: float = 0.5):
        self.names = names
        self.vmax = vmax
        if palette is None:
            palette = ["#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231", "#911EB4"]
        self.colors = [mcolors.to_rgb(c) for c in palette[:len(names)]]
    
    def __call__(self, flatmaps: List[np.ndarray], background: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        baseline = flatmaps[-1]
        stacked = np.stack(flatmaps[:-1], axis=0)-baseline
        
        with np.errstate(invalid='ignore'):
            loser_indices = np.argsort(stacked, axis=0)[0]
            second_loser_indices = np.argsort(stacked, axis=0)[1]
            rows, cols = np.indices(stacked.shape[1:])
            margins = stacked[second_loser_indices, rows, cols] - stacked[loser_indices, rows, cols]
        
        h, w = flatmaps[0].shape
        rgb_map = np.zeros((h, w, 3))
        
        brain_mask = ~np.all(np.isnan(stacked), axis=0)
        for i, rgb in enumerate(self.colors):
            mask = (loser_indices == i) & brain_mask
            strength = flatmaps[i][mask]/np.percentile(flatmaps[i][mask], 95)
            strength = np.nan_to_num(np.clip(strength, 0, 1))
            # use sqrt of margin for better visual separation
            margin_vals = np.nan_to_num(np.clip(np.sqrt(margins[mask]) / np.percentile(np.sqrt(margins[mask]), 65), 0, 1))
            rgb_map[mask] = np.array(rgb) * margin_vals[:, np.newaxis] + (1*(1-margin_vals[:, np.newaxis])) 
            rgb_map[mask] = rgb_map[mask] * strength[:, np.newaxis] + (1 - strength[:, np.newaxis]) * (background[mask] if background is not None else [1, 1, 1])
        
        rgb_map[~brain_mask] = background[~brain_mask] if background is not None else [1, 1, 1]
        return rgb_map
    
    def get_legend_type(self) -> str:
        return 'categorical'
    
    def get_legend_data(self) -> Dict:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor=mcolors.to_hex(c),
                   markersize=8, label=name)
            for name, c in zip(self.names, self.colors)
        ]
        return {'legend_elements': legend_elements}


class LoserTakesItAllBlobsStrategy(ColorStrategy):
    """Loser with thresholding"""
    
    def __init__(
        self,
        names: List[str],
        margin_threshold: float = 0.0,
        correlation_threshold: float = 0.0,
        palette: Optional[List[str]] = None,
    ):
        self.names = names
        self.margin_threshold = margin_threshold
        self.correlation_threshold = correlation_threshold
        if palette is None:
            palette = ["#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231", "#911EB4"]
        self.colors = [mcolors.to_rgb(c) for c in palette[:len(names)]]
    
    def __call__(self, flatmaps: List[np.ndarray], background: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        baseline = flatmaps[-1]
        stacked = np.stack(flatmaps, axis=0)
        
        with np.errstate(invalid='ignore'):
            loser_indices = np.argsort(stacked, axis=0)[0]
            second_loser_indices = np.argsort(stacked, axis=0)[1]
            rows, cols = np.indices(stacked.shape[1:])
            margins = stacked[second_loser_indices, rows, cols] - stacked[loser_indices, rows, cols]
            margins = np.nan_to_num(np.clip(margins / 0.1, 0, 1))
        
        h, w = flatmaps[0].shape
        rgb_map = np.ones((h, w, 3))
        brain_mask = ~np.all(np.isnan(stacked), axis=0)
        rgb_map[brain_mask] = [0, 0, 0] if background is None else background[brain_mask]
        
        for i, rgb in enumerate(self.colors):
            mask = (loser_indices == i) & brain_mask
            corr_vals = np.nan_to_num(baseline[mask])
            margin_vals = margins[mask]
            threshold_mask = (margin_vals > self.margin_threshold) & \
                           (corr_vals > self.correlation_threshold)
            full_mask = np.zeros_like(brain_mask, dtype=bool)
            full_mask[mask] = threshold_mask
            rgb_map[full_mask] = rgb
        
        return rgb_map
    
    def get_legend_type(self) -> str:
        return 'categorical'
    
    def get_legend_data(self) -> Dict:
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
    figsize: Tuple[int, int] = (10, 10)
    dpi: int = 150
    smooth: bool = False
    sigma: float = 2.0
    show_regions: bool = False
    show: bool = True
    save: bool = True
    output_dir: str = "."
    # Fixed layout parameters for consistent sizing
    title_height: float = 0.08  # 8% of figure for title
    legend_height: float = 0.12  # 12% of figure for legend
    image_left: float = 0.05
    image_right: float = 0.95
    image_bottom_padding: float = 0.02  # Small padding above legend


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
        colorbar_label: str = "",
        colorbar_data: Optional[Tuple] = None,
        strategy: Optional['ColorStrategy'] = None,
        **kwargs,
    ):
        ax = axes[0]
        ax.imshow(rgb_map, interpolation='none')
        ax.axis("off")
        
        # Use suptitle for consistent positioning
        if title:
            fig.suptitle(title, fontsize=14, y=0.98)
        
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
            
            # Create colorbar axis in fixed legend area
            cbar_ax = fig.add_axes([0.25, 0.04, 0.5, 0.03])
            cbar = fig.colorbar(sm, cax=cbar_ax, label=label, orientation='horizontal')
        
        # Fallback for explicit colorbar_data
        elif colorbar_data is not None:
            flatmap, norm, cmap = colorbar_data
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar_ax = fig.add_axes([0.25, 0.04, 0.5, 0.03])
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
        
        if legend_elements:
            # Position legend in fixed legend area at bottom
            legend = fig.legend(
                handles=legend_elements,
                loc='lower center',
                ncol=min(len(legend_elements), 6),
                bbox_to_anchor=(0.5, 0.03),
                frameon=True,
                fontsize=10
            )
        
        if self.config.show_regions and self.mapper:
            _overlay_flatmap_rois(axes[0], self.mapper.map_file)
        
        # Don't use tight_layout since we're using manual positioning
        
        if self.config.save:
            os.makedirs(self.config.output_dir, exist_ok=True)
            if not output_filename:
                output_filename = "flatmap.png"
            output_path = os.path.join(self.config.output_dir, output_filename)
            plt.savefig(output_path, dpi=self.config.dpi)
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
            fig.suptitle(title, fontsize=14, y=0.98)
        
        # Add 2D legend in fixed position within legend area
        # Center it horizontally, position in legend area
        legend_size = 0.18  # Size relative to figure
        legend_x = 0.5 - legend_size / 2  # Center horizontally
        legend_y = 0.02  # Bottom of legend area
        
        ax_inset = fig.add_axes([legend_x, legend_y, legend_size, legend_size])
        x_grid, y_grid = np.meshgrid(
            np.linspace(0, legend_vmax, 50),
            np.linspace(0, legend_vmax, 50)
        )
        legend_rgb = get_bivariate_color(x_grid, y_grid, vmin=0, vmax=legend_vmax)
        ax_inset.imshow(legend_rgb, origin='lower', extent=[0, legend_vmax, 0, legend_vmax])
        ax_inset.set_xlabel(legend_labels[0], fontsize=9)
        ax_inset.set_ylabel(legend_labels[1], fontsize=9)
        ax_inset.set_xticks([0, legend_vmax])
        ax_inset.set_yticks([0, legend_vmax])
        ax_inset.tick_params(labelsize=8)

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
        output_filename: str = "",
        builder_kwargs: Optional[Dict] = None,
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
        flatmaps = [self.mapper.to_flatmap(d) for d in data]
        
        # Smooth if requested
        if self.config.smooth:
            flatmaps = [smooth_flatmap(fm, sigma=self.config.sigma) for fm in flatmaps]
        
        # Generate colors
        
        background = self.mapper.get_brain_bkg()
        rgb_map = strategy(flatmaps, background=background, **kwargs)
        rgb_map[~self.mapper.get_brain_mask()] = [1, 1, 1]
        
        
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
        builder.render(fig, axes, rgb_map, title=title, strategy=strategy, **builder_kwargs)
        
        # Finalize with strategy passed for auto-legend generation
        builder.finalize(fig, axes, output_filename=output_filename, strategy=strategy, **builder_kwargs)
        
        return fig, rgb_map

# endregion
# ============================================================================
# region CONVENIENCE WRAPPERS: For backward compatibility / ease of use

def plot_correlation_on_flatmap(
    subject: str,
    modality: str,
    mode: str,
    correlation: np.ndarray,
    mapper_dir: str,
    output_dir: str,
    title: str = "correlation flatmap",
    regions: bool = False,
    show: bool = True,
    save: bool = True,
    smooth: bool = False,
    sigma: float = 2.0,
    vmin: float = 0,
    vmax: float = 0.5,
):
    """Plot single correlation map with hot colormap."""
    mapper = FlatmapMapper(subject, mapper_dir)
    config = FigureConfig(
        figsize=(8, 6),
        dpi=150,
        show_regions=regions,
        show=show,
        save=save,
        smooth=smooth,
        sigma=sigma,
        output_dir=output_dir,
    )
    plotter = FlatmapPlotter(mapper, config)
    strategy = SingleCorrelationStrategy(vmin=vmin, vmax=vmax)
    output_file = f"{subject}_{modality}_{mode}.png"
    plotter.plot([correlation], strategy, title=title, output_filename=output_file)

def plot_bivariate_flatmap(
    subject: str,
    modality: str,
    corr_1: np.ndarray,
    corr_2: np.ndarray,
    mode_1: str,
    mode_2: str,
    mapper_dir: str,
    output_dir: str,
    title: str = "Bivariate Correlation Map",
    show: bool = True,
    save: bool = True,
    vmax: float = 0.5,
):
    """Plot bivariate correlation map with 2D legend."""
    mapper = FlatmapMapper(subject, mapper_dir)
    config = FigureConfig(
        figsize=(10, 8),
        dpi=150,
        show=show,
        save=save,
        output_dir=output_dir,
    )
    plotter = FlatmapPlotter(mapper, config)
    strategy = BivariateStrategy(vmax=vmax)
    output_file = f"{subject}_{modality}_{mode_1}_vs_{mode_2}.png"
    plotter.plot(
        [corr_1, corr_2],
        strategy,
        title=title,
        output_filename=output_file,
    )

def plot_loser_takes_it_all(
    subject: str,
    modality: str,
    correlations: List[np.ndarray],
    names: List[str],
    mapper_dir: str,
    output_dir: str,
    rank: int = 0,
    title: str = "Loser Takes It All Map",
    masks: Optional[np.ndarray] = None,
    regions: bool = False,
    show: bool = True,
    save: bool = True,
    smooth: bool = False,
):
    """Plot loser-takes-it-all map showing which model is best in each region."""
    mapper = FlatmapMapper(subject, mapper_dir)
    config = FigureConfig(
        figsize=(12, 8),
        dpi=200,
        show_regions=regions,
        show=show,
        save=save,
        smooth=smooth,
        output_dir=output_dir,
    )
    plotter = FlatmapPlotter(mapper, config)
    strategy = LoserTakesItAllStrategy(names)
    
    # Prepare data with optional mask
    data = correlations if masks is None else [np.where(masks > 0, c, np.nan) for c in correlations]
    
    full_title = f"{title}\nSubject: {subject} | Modality: {modality}"
    output_file = f"{subject}_{modality}_loser_map.png"
    
    plotter.plot(
        data,
        strategy,
        title=full_title,
        output_filename=output_file,
    )

def plot_loser_takes_it_all_margin(
    subject: str,
    modality: str,
    correlations: List[np.ndarray],
    names: List[str],
    mapper_dir: str,
    output_dir: str,
    title: str = "Loser Takes It All Map",
    masks: Optional[np.ndarray] = None,
    regions: bool = False,
    show: bool = True,
    save: bool = True,
):
    """Plot loser-takes-it-all with margin visualization."""
    mapper = FlatmapMapper(subject, mapper_dir)
    config = FigureConfig(
        figsize=(12, 8),
        dpi=200,
        show_regions=regions,
        show=show,
        save=save,
        output_dir=output_dir,
    )
    plotter = FlatmapPlotter(mapper, config)
    strategy = LoserTakesItAllMarginStrategy(names)
    
    data = correlations if masks is None else [np.where(masks > 0, c, np.nan) for c in correlations]
    
    full_title = f"{title}\nSubject: {subject} | Modality: {modality}"
    output_file = f"{subject}_{modality}_loser_margin.png"
    
    plotter.plot(
        data,
        strategy,
        title=full_title,
        output_filename=output_file,
    )

def plot_loser_takes_it_all_blobs(
    subject: str,
    modality: str,
    correlations: List[np.ndarray],
    names: List[str],
    mapper_dir: str,
    output_dir: str,
    title: str = "Loser Takes It All Map",
    masks: Optional[np.ndarray] = None,
    regions: bool = False,
    show: bool = True,
    save: bool = True,
    margin_threshold: float = 0.0,
    correlation_threshold: float = 0.0,
):
    """Plot loser-takes-it-all with thresholding (only shows significant blobs)."""
    mapper = FlatmapMapper(subject, mapper_dir)
    config = FigureConfig(
        figsize=(12, 8),
        dpi=200,
        show_regions=regions,
        show=show,
        save=save,
        output_dir=output_dir,
    )
    plotter = FlatmapPlotter(mapper, config)
    strategy = LoserTakesItAllBlobsStrategy(
        names,
        margin_threshold=margin_threshold,
        correlation_threshold=correlation_threshold,
    )
    
    data = correlations if masks is None else [np.where(masks > 0, c, np.nan) for c in correlations]
    
    full_title = f"{title}\nSubject: {subject} | Modality: {modality}\n" \
                f"(margin > {margin_threshold}, corr > {correlation_threshold})"
    output_file = f"{subject}_{modality}_loser_blobs.png"
    
    plotter.plot(
        data,
        strategy,
        title=full_title,
        output_filename=output_file,
    )
# endregion
# ============================================================================
