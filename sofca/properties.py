# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 15:35:27 2026

@author: s.gupta
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from skimage import measure
from skimage.measure import regionprops
from skimage.measure import marching_cubes, mesh_surface_area
from scipy.spatial import ConvexHull, QhullError


def grain_contact_percent(
    grain_labels: np.ndarray,
    phase_labels: np.ndarray,
    voxel_size_um: float,
    include_volume_border: bool = True
) -> pd.DataFrame:
    """
    Computes exposed surface area and contact percentages of each grain
    with ALL phase labels present in phase_labels.
    Uses voxel-face counting (6-neighborhood).
    """
    if grain_labels.shape != phase_labels.shape:
        raise ValueError("grain_labels and phase_labels must have same shape")

    max_grain = int(grain_labels.max())
    max_phase = int(phase_labels.max())

    if max_grain == 0:
        return pd.DataFrame()

    total_faces = np.zeros(max_grain + 1, dtype=np.int64)
    contact_faces = np.zeros((max_grain + 1, max_phase + 1), dtype=np.int64)

    def _accumulate(g, g_nbr, ph_nbr):
        surf = (g > 0) & (g != g_nbr)
        if not np.any(surf):
            return

        ids = g[surf].ravel().astype(np.int64)
        phs = ph_nbr[surf].ravel().astype(np.int64)

        total_faces[:] += np.bincount(ids, minlength=max_grain + 1)
        np.add.at(contact_faces, (ids, phs), 1)

    directions = [
        (grain_labels[:, :, :-1], grain_labels[:, :,  1:], phase_labels[:, :,  1:]),  # +X
        (grain_labels[:, :,  1:], grain_labels[:, :, :-1], phase_labels[:, :, :-1]),  # -X
        (grain_labels[:, :-1, :], grain_labels[:,  1:, :], phase_labels[:,  1:, :]),  # +Y
        (grain_labels[:,  1:, :], grain_labels[:, :-1, :], phase_labels[:, :-1, :]),  # -Y
        (grain_labels[:-1, :, :], grain_labels[ 1:, :, :], phase_labels[ 1:, :, :]),  # +Z
        (grain_labels[ 1:, :, :], grain_labels[:-1, :, :], phase_labels[:-1, :, :])   # -Z
    ]

    for g, g_nbr, ph_nbr in tqdm(directions, desc="Counting grain contact faces", unit="direction"):
        _accumulate(g, g_nbr, ph_nbr)

    if include_volume_border:
        border_slabs = [
            grain_labels[:, :, 0],
            grain_labels[:, :, -1],
            grain_labels[:, 0, :],
            grain_labels[:, -1, :],
            grain_labels[0, :, :],
            grain_labels[-1, :, :]
        ]
        for slab in tqdm(border_slabs, desc="Processing volume borders", unit="face"):
            m = slab > 0
            if np.any(m):
                ids = slab[m].ravel().astype(np.int64)
                total_faces[:] += np.bincount(ids, minlength=max_grain + 1)

    grain_ids = np.unique(grain_labels)
    grain_ids = grain_ids[grain_ids != 0].astype(int)

    face_area_um2 = float(voxel_size_um) ** 2

    df = pd.DataFrame(index=grain_ids)
    df.index.name = "grain_id"

    df["total_exposed_faces"] = total_faces[grain_ids]
    df["total_surface_area_um2"] = df["total_exposed_faces"] * face_area_um2

    denom = df["total_exposed_faces"].to_numpy(dtype=float)
    valid = denom > 0

    phase_vals = np.unique(phase_labels)
    phase_vals = phase_vals[phase_vals != 0].astype(int)

    for ph in phase_vals:
        faces = contact_faces[grain_ids, ph]
        df[f"contact_phase_{ph}_faces"] = faces
        df[f"contact_phase_{ph}_area_um2"] = faces * face_area_um2

        pct = np.zeros_like(denom)
        pct[valid] = 100.0 * (faces[valid] / denom[valid])
        df[f"contact_phase_{ph}_percent"] = pct

    return df


def calculate_properties(labelled_image, phase_labels, Properties, voxel_size_um, step_size):
    """
    Your original calculate_properties, with:
      - Surface area via marching cubes
      - regionprops_table
      - sphericity
      - Feret diameters (if requested)
      - contact stats via grain_contact_percent
    """
    voxel_size_um = float(voxel_size_um)
    unique_labels = np.unique(labelled_image)
    unique_labels = unique_labels[unique_labels != 0]

    def calculate_surface_area(label, labelled_image, step_size):
        non_zero_indices = np.argwhere(labelled_image == label)
        if non_zero_indices.shape[0] < 2:
            return 0.0

        min_idx = np.maximum(non_zero_indices.min(axis=0) - 2, 0)
        max_idx = np.minimum(non_zero_indices.max(axis=0) + 2, np.array(labelled_image.shape) - 1)

        region = labelled_image[min_idx[0]:max_idx[0]+1,
                                min_idx[1]:max_idx[1]+1,
                                min_idx[2]:max_idx[2]+1]

        mask = (region == label).astype(np.uint8)
        if np.any(np.array(mask.shape) < 2) or mask.sum() == 0:
            return 0.0

        verts, faces, *_ = marching_cubes(mask, level=0.5, spacing=(voxel_size_um,) * 3, step_size=step_size)
        return mesh_surface_area(verts, faces)

    def calculate_surface_areas(labelled_image, step_size, labels_to_process):
        return pd.DataFrame({
            'label': labels_to_process,
            'Surface Area': Parallel(n_jobs=-1)(
                delayed(calculate_surface_area)(lab, labelled_image, step_size)
                for lab in tqdm(labels_to_process, desc='Calculating Surface Areas')
            )
        })

    surface_areas_df = calculate_surface_areas(labelled_image, step_size, unique_labels)

    feret_keys = {'min_feret_diameter', 'max_feret_diameter'}
    feret_requested = feret_keys.intersection(Properties)
    Properties = [p for p in Properties if p not in feret_keys]

    regionprops_df = pd.DataFrame(
        measure.regionprops_table(labelled_image, intensity_image=phase_labels, properties=Properties)
    ).astype({'label': int}).set_index('label')

    rename_map = {}
    if 'area' in regionprops_df.columns:
        rename_map['area'] = 'Volume'
    if 'bbox_area' in regionprops_df.columns:
        rename_map['bbox_area'] = 'Bounding_Box_Volume'
    if 'filled_area' in regionprops_df.columns:
        rename_map['filled_area'] = 'Filled_Volume'
    regionprops_df.rename(columns=rename_map, inplace=True)

    surface_areas_df['label'] = surface_areas_df['label'].astype(int)
    merged = surface_areas_df.merge(regionprops_df, how='left', on='label')

    voxel_volume = voxel_size_um ** 3

    length_keys = {'equivalent_diameter', 'major_axis_length', 'minor_axis_length'}
    volume_keys = {'Volume', 'Bounding_Box_Volume', 'Filled_Volume'}

    for col in merged.columns:
        if col in length_keys:
            merged[col] *= voxel_size_um
        elif col in volume_keys:
            merged[col] *= voxel_volume

    merged['Sphericity'] = np.nan
    valid_area = merged['Surface Area'] > 0
    merged.loc[valid_area, 'Sphericity'] = ((np.pi ** (1/3)) *
        (6 * merged.loc[valid_area, 'Volume']) ** (2/3)) / (merged.loc[valid_area, 'Surface Area'])

    if feret_requested:

        def fibonacci_sphere_samples(n=64800):
            indices = np.arange(n, dtype=float) + 0.5
            phi = np.arccos(1 - 2 * indices / n)
            theta = np.pi * (1 + 5**0.5) * indices
            return np.stack([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ], axis=1)

        def exact_feret_diameters(coords):
            if len(coords) < 2:
                return 0.0, 0.0
            try:
                hull = ConvexHull(coords)
                pts = coords[hull.vertices]
            except (QhullError, ValueError):
                pts = coords

            directions = fibonacci_sphere_samples()
            projections = pts @ directions.T
            spans = np.max(projections, axis=0) - np.min(projections, axis=0)
            return np.max(spans), np.min(spans)

        def _process_object(prop):
            mask = prop.image
            if mask.sum() == 0:
                return prop.label, 0.0, 0.0

            coords = np.argwhere(mask)
            min_z, min_y, min_x, _, _, _ = prop.bbox
            global_coords = coords + np.array([min_z, min_y, min_x])

            max_feret, min_feret = exact_feret_diameters(global_coords)
            return prop.label, max_feret, min_feret

        def compute_feret_diameters(label_img, n_jobs=-1):
            props = regionprops(label_img)
            results = Parallel(n_jobs=n_jobs)(
                delayed(_process_object)(prop)
                for prop in tqdm(props, desc="Computing Feret diameters")
            )
            return pd.DataFrame(results, columns=['label', 'Max_Feret', 'Min_Feret']).set_index('label')

        feret_df = compute_feret_diameters(labelled_image)
        merged = merged.merge(feret_df, how='left', on='label')

        merged['Max_Feret'] *= voxel_size_um
        merged['Min_Feret'] *= voxel_size_um
        merged['Feret_ratio'] = merged['Min_Feret'] / merged['Max_Feret']

    if 'Bounding_Box_Volume' in merged.columns and 'Volume' in merged.columns:
        merged['Volume/bbox'] = merged['Volume'] / merged['Bounding_Box_Volume']

    merged = merged.rename(columns={'label': 'Label'}).set_index('Label')

    contact_df = grain_contact_percent(
        grain_labels=labelled_image,
        phase_labels=phase_labels,
        voxel_size_um=voxel_size_um,
        include_volume_border=True
    )

    merged = merged.merge(contact_df, how="left", left_index=True, right_index=True)
    return merged


def tpb_contact_length_per_grain(grain_labels, tpb_x, tpb_y, tpb_z, voxel_size_um):
    """
    TPB contact length per grain (NO splitting)
    For each TPB edge, count it for every distinct non-zero grain ID among the
    4 incident voxels. (This matches "grain surface length in contact with TPB".)

    Returns a DataFrame indexed by grain_id with per-direction edge counts and lengths.
    """
    max_grain = int(grain_labels.max())
    if max_grain == 0:
        return pd.DataFrame()

    voxel_size_um = float(voxel_size_um)

    def count_contact_edges(tpb_mask, g0, g1, g2, g3):
        """
        For each True position in tpb_mask, look at the 4 incident labels (g0..g3),
        and add +1 to every distinct non-zero grain id present.
        """
        if not np.any(tpb_mask):
            return np.zeros(max_grain + 1, dtype=np.int64)

        a = g0[tpb_mask].ravel().astype(np.int64)
        b = g1[tpb_mask].ravel().astype(np.int64)
        c = g2[tpb_mask].ravel().astype(np.int64)
        d = g3[tpb_mask].ravel().astype(np.int64)

        # Keep only non-zero IDs, and avoid counting the same grain twice within one edge
        m0 = (a > 0)
        m1 = (b > 0) & (b != a)
        m2 = (c > 0) & (c != a) & (c != b)
        m3 = (d > 0) & (d != a) & (d != b) & (d != c)

        ids = np.concatenate([a[m0], b[m1], c[m2], d[m3]])
        if ids.size == 0:
            return np.zeros(max_grain + 1, dtype=np.int64)

        return np.bincount(ids, minlength=max_grain + 1)

    # X-oriented edges: incident voxels are a 2x2 in (y,x) at two z planes? (as you had)
    cx = count_contact_edges(
        tpb_x,
        grain_labels[:-1, :-1, :],
        grain_labels[ 1:, :-1, :],
        grain_labels[:-1,  1:, :],
        grain_labels[ 1:,  1:, :]
    )

    cy = count_contact_edges(
        tpb_y,
        grain_labels[:-1, :, :-1],
        grain_labels[ 1:, :, :-1],
        grain_labels[:-1, :,  1:],
        grain_labels[ 1:, :,  1:]
    )

    cz = count_contact_edges(
        tpb_z,
        grain_labels[:, :-1, :-1],
        grain_labels[:,  1:, :-1],
        grain_labels[:, :-1,  1:],
        grain_labels[:,  1:,  1:]
    )

    grain_ids = np.unique(grain_labels)
    grain_ids = grain_ids[grain_ids != 0].astype(int)

    df = pd.DataFrame(index=grain_ids)
    df.index.name = "grain_id"

    df["tpb_edges_x_contact_count"] = cx[grain_ids]
    df["tpb_edges_y_contact_count"] = cy[grain_ids]
    df["tpb_edges_z_contact_count"] = cz[grain_ids]

    df["tpb_contact_length_x_um"] = df["tpb_edges_x_contact_count"] * voxel_size_um
    df["tpb_contact_length_y_um"] = df["tpb_edges_y_contact_count"] * voxel_size_um
    df["tpb_contact_length_z_um"] = df["tpb_edges_z_contact_count"] * voxel_size_um

    df["tpb_contact_length_total_um"] = (
        df["tpb_edges_x_contact_count"]
        + df["tpb_edges_y_contact_count"]
        + df["tpb_edges_z_contact_count"]
    ) * voxel_size_um

    return df
