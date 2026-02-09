# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 15:33:48 2026

@author: s.gupta
"""

import os
import gc
import numpy as np
import pandas as pd


def tpb_edge_maps(labels: np.ndarray, Ni_label: int, YSZ_label: int, Pore_label: int):
    """
    Returns:
      tpb_x: (Z-1, Y-1, X)   edges parallel to X
      tpb_y: (Z-1, Y,   X-1) edges parallel to Y
      tpb_z: (Z,   Y-1, X-1) edges parallel to Z

    Edge is TPB if its 4 incident voxels contain {Ni,YSZ,Pore}.
    """
    # X-edges
    a = labels[:-1, :-1, :]
    b = labels[ 1:, :-1, :]
    c = labels[:-1,  1:, :]
    d = labels[ 1:,  1:, :]
    has_N = (a == Ni_label)  | (b == Ni_label)  | (c == Ni_label)  | (d == Ni_label)
    has_Y = (a == YSZ_label) | (b == YSZ_label) | (c == YSZ_label) | (d == YSZ_label)
    has_P = (a == Pore_label)| (b == Pore_label)| (c == Pore_label)| (d == Pore_label)
    tpb_x = has_N & has_Y & has_P
    del a, b, c, d, has_N, has_Y, has_P
    gc.collect()

    # Y-edges
    a = labels[:-1, :, :-1]
    b = labels[ 1:, :, :-1]
    c = labels[:-1, :,  1:]
    d = labels[ 1:, :,  1:]
    has_N = (a == Ni_label)  | (b == Ni_label)  | (c == Ni_label)  | (d == Ni_label)
    has_Y = (a == YSZ_label) | (b == YSZ_label) | (c == YSZ_label) | (d == YSZ_label)
    has_P = (a == Pore_label)| (b == Pore_label)| (c == Pore_label)| (d == Pore_label)
    tpb_y = has_N & has_Y & has_P
    del a, b, c, d, has_N, has_Y, has_P
    gc.collect()

    # Z-edges
    a = labels[:, :-1, :-1]
    b = labels[:,  1:, :-1]
    c = labels[:, :-1,  1:]
    d = labels[:,  1:,  1:]
    has_N = (a == Ni_label)  | (b == Ni_label)  | (c == Ni_label)  | (d == Ni_label)
    has_Y = (a == YSZ_label) | (b == YSZ_label) | (c == YSZ_label) | (d == YSZ_label)
    has_P = (a == Pore_label)| (b == Pore_label)| (c == Pore_label)| (d == Pore_label)
    tpb_z = has_N & has_Y & has_P
    del a, b, c, d, has_N, has_Y, has_P
    gc.collect()

    return tpb_x, tpb_y, tpb_z


def voxelize_all_tpb_edges(labels: np.ndarray, tpb_x: np.ndarray, tpb_y: np.ndarray, tpb_z: np.ndarray) -> np.ndarray:
    """
    Convert edge-based TPB into ONE voxel mask (Z,Y,X) for visualization:
    voxel=True if any TPB edge touches that voxel.
    """
    Z, Y, X = labels.shape
    vox = np.zeros((Z, Y, X), dtype=bool)

    # X edges touch 4 voxels
    vox[:-1, :-1, :] |= tpb_x
    vox[ 1:, :-1, :] |= tpb_x
    vox[:-1,  1:, :] |= tpb_x
    vox[ 1:,  1:, :] |= tpb_x

    # Y edges
    vox[:-1, :, :-1] |= tpb_y
    vox[ 1:, :, :-1] |= tpb_y
    vox[:-1, :,  1:] |= tpb_y
    vox[ 1:, :,  1:] |= tpb_y

    # Z edges
    vox[:, :-1, :-1] |= tpb_z
    vox[:,  1:, :-1] |= tpb_z
    vox[:, :-1,  1:] |= tpb_z
    vox[:,  1:,  1:] |= tpb_z

    gc.collect()
    return vox


def TPB_extraction(
    labels: np.ndarray,
    voxel_size_um: float,
    Ni_label: int = 3,
    YSZ_label: int = 2,
    Pore_label: int = 1,
    Path_to_save: str | None = None,
    excel_filename: str = "tpb_results.xlsx",
    save_excel: bool = False
):
    """
    Returns:
      results_df, tpb_voxel_mask, tpb_x, tpb_y, tpb_z

    If save_excel=True and Path_to_save provided, writes an Excel file.
    """
    voxel_size_um = float(voxel_size_um)

    tpb_x, tpb_y, tpb_z = tpb_edge_maps(labels, Ni_label, YSZ_label, Pore_label)

    Nx = int(tpb_x.sum())
    Ny = int(tpb_y.sum())
    Nz = int(tpb_z.sum())

    TPB_length_um = (Nx + Ny + Nz) * voxel_size_um
    total_volume_um3 = float(labels.size) * (voxel_size_um ** 3)
    TPB_density_um2 = TPB_length_um / total_volume_um3

    tpb_voxel_mask = voxelize_all_tpb_edges(labels, tpb_x, tpb_y, tpb_z)

    results_df = pd.DataFrame([{
        "TPB_length_um": TPB_length_um,
        "TPB_density_um^-2": TPB_density_um2,
        "TPB_edges_x_count": Nx,
        "TPB_edges_y_count": Ny,
        "TPB_edges_z_count": Nz,
        "voxel_size_um": voxel_size_um,
        "volume_shape_ZYX": str(labels.shape),
        "Ni_label": int(Ni_label),
        "YSZ_label": int(YSZ_label),
        "Pore_label": int(Pore_label),
        "tpb_definition": "edge-based: edge is TPB if its 4 incident voxels contain {Ni,YSZ,Pore}",
        "voxelized_output_note": "voxel mask is for visualization; length/density from edge counts"
    }])

    if save_excel and Path_to_save is not None:
        os.makedirs(Path_to_save, exist_ok=True)
        out_xlsx = os.path.join(Path_to_save, excel_filename)
        results_df.to_excel(out_xlsx, index=False)

    gc.collect()
    return results_df, tpb_voxel_mask, tpb_x, tpb_y, tpb_z
