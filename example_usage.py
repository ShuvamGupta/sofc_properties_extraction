# -*- coding: utf-8 -*-
"""
example_usage.py

Demonstration of SOFCA workflow for:

- Grain labeling
- Cleaning labeled grains
- Total TPB extraction
- Grain-resolved TPB calculation
- Geometric property extraction
"""
import sofca 


# ---------------------------------------------------------------------
# 1. LOAD SEGMENTED DATA
# ---------------------------------------------------------------------

# sofca.upload_images(image_path: str) -> np.ndarray
# Uploads the segmented image mask (2D or 3D).

Segmented = sofca.upload_images(r"C:\Users\s.gupta\Downloads\Test\Segmented_36")

# ---------------------------------------------------------------------
# 2. DEFINE PHASE LABEL VALUES
# ---------------------------------------------------------------------
ni_mask   = 2
ysz_mask  = 1
pore_mask = 0


# ---------------------------------------------------------------------
# 3. LABEL GRAINS (CONNECTED COMPONENT ANALYSIS)
# ---------------------------------------------------------------------

# sofca.Labelling_phase_grains(
#     labels: np.ndarray,
#     Ni_label: int,
#     YSZ_label: int,
#     Pore_label: int
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
#
# Returns:
#   Ni_labels, YSZ_labels, Pore_labels

Ni_labels, ysz_labels, pores_labels = sofca.Labelling_phase_grains(Segmented, ni_mask, ysz_mask, pore_mask)


# ---------------------------------------------------------------------
# 4. REMOVE BORDER-CONNECTED GRAINS
# ---------------------------------------------------------------------

# sofca.delete_border_labels(
#     grain_labelled_phase_image: np.ndarray
# ) -> np.ndarray
#
# Removes grains touching the domain boundary.

Ni_labels = sofca.delete_border_labels(Ni_labels)
ysz_labels = sofca.delete_border_labels(ysz_labels)
pores_labels = sofca.delete_border_labels(pores_labels)

# ---------------------------------------------------------------------
# 5. REMOVE SMALL PARTICLES
# ---------------------------------------------------------------------

# Size threshold in micrometers

Size_threshold = 0.01

# sofca.delete_small_particles(
#     grain_labelled_phase_image: np.ndarray,
#     size_threshold_um: float
# ) -> np.ndarray
#
# Removes particles smaller than threshold.

Ni_labels = sofca.delete_small_particles(Ni_labels, Size_threshold)
ysz_labels = sofca.delete_small_particles(ysz_labels, Size_threshold)
pore_labels = sofca.delete_small_particles(pores_labels, Size_threshold)


# ---------------------------------------------------------------------
# 6. TOTAL TPB EXTRACTION
# ---------------------------------------------------------------------

# sofca.TPB_extraction(
#     segmented_image: np.ndarray,
#     voxel_size_um: float,
#     Ni_label: int,
#     YSZ_label: int,
#     Pore_label: int,
#     output_path: str,
#     output_filename: str,
#     save_excel: bool = bool
# ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
#
# Returns:
#   results_df       -> summary statistics
#   tpb_voxel_mask   -> TPB voxel mask
#   tpb_x, tpb_y, tpb_z -> TPB edge masks in x, y, z directions

results_df, tpb_voxel_mask, tpb_x, tpb_y, tpb_z = sofca.TPB_extraction(Segmented,0.1,2,1,0, r"C:\Users\s.gupta\Downloads\Test","tpb_results.xlsx",True)

# ---------------------------------------------------------------------
# 7. TPB LENGTH PER NI GRAIN
# ---------------------------------------------------------------------

# sofca.tpb_contact_length_per_grain(
#     grain_labelled_phase_image: np.ndarray,
#     tpb_x: np.ndarray,
#     tpb_y: np.ndarray,
#     tpb_z: np.ndarray,
#     voxel_size_um: float
# ) -> pd.DataFrame
#
# Computes TPB length shared by each Ni grain.

TPB_per_grain = sofca.tpb_contact_length_per_grain(Ni_labels, tpb_x, tpb_y, tpb_z, 0.01) 


# ---------------------------------------------------------------------
# 8. GEOMETRIC PROPERTIES OF NI GRAINS
# ---------------------------------------------------------------------

# List of region properties to extract

Properties = [
    'label', 'area', 'min_intensity', 'max_intensity', 'equivalent_diameter',
    'mean_intensity', 'bbox_area', 'filled_area',
    'min_feret_diameter', 'max_feret_diameter']


# sofca.calculate_properties(
#     grain_labelled_phase_image: np.ndarray,
#     segmented_image: np.ndarray,
#     properties: List[str],
#     voxel_size_um: float,
#     stepsize for mesh to calculate surface area: int
# ) -> pd.DataFrame
#
# Returns geometric property table for grains.
Properties = sofca.calculate_properties(Ni_labels, Segmented, Properties, 0.01, 1)


# Plot distributions
"""
    Plots the distribution of a selected property from a microstructure dataset.

    This function takes a DataFrame of grain/particle properties (like the one returned
    by `sofca.calculate_properties`) and visualizes the distribution of a specified
    column. Each particle can be weighted by another property (e.g., volume), and the 
    plot can be a histogram or a cumulative distribution.

    Parameters
    ----------
    Properties : pd.DataFrame
        Table of particle/grain properties, indexed by grain ID.
    x_column : str
        Column name to plot on the x-axis (e.g., 'equivalent_diameter').
    weight_column : str, optional
        Column to weight each particle by (e.g., 'Volume'). Default is None.
    kind : str, optional
        Type of plot:
        - 'hist' → histogram
        - 'cumulative' → cumulative distribution
        Default is 'hist'.
    metric : str, optional
        Metric for the distribution:
        - 'number' → counts particles (weighted if weight_column is set)
        - 'volume' → fraction of total volume
        Default is 'number'.
    bins : int, optional
        Number of bins for histogram. Default is 30.

    Returns
    -------
    None
        Displays a plot of the distribution. Does not return a DataFrame.
    
    Example
    -------
    sofca.plot_distribution(
        Properties,
        x_column='equivalent_diameter',
        weight_column='Volume',
        kind='hist',
        metric='number',
        bins=30
    )
"""
    
sofca.plot_distribution(Properties, x_column='equivalent_diameter', weight_column = 'Volume', kind='hist', metric='number', bins=30)


