# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 15:32:19 2026

@author: s.gupta
"""

import numpy as np
from skimage.measure import label as sk_label
from skimage import morphology


def Labelling_phase_grains(labels: np.ndarray, Ni_label: int, YSZ_label: int, Pore_label: int):
    """
    Post-TPB routine:
    - Computes 26-connected components for Ni / YSZ / Pore
    - Returns labeled component volumes (NO disk I/O)
    """
    ni_mask   = (labels == Ni_label)
    ysz_mask  = (labels == YSZ_label)
    pore_mask = (labels == Pore_label)

    ni_cc26   = sk_label(ni_mask,   connectivity=3)
    ysz_cc26  = sk_label(ysz_mask,  connectivity=3)
    pore_cc26 = sk_label(pore_mask, connectivity=3)

    return ni_cc26, ysz_cc26, pore_cc26


def delete_small_particles(labelled_image: np.ndarray, size_threshold_vox: int) -> np.ndarray:
    """
    Removes labeled particles smaller than size_threshold_vox voxels.
    Threshold is in VOXELS (not microns).
    """
    if size_threshold_vox is None or int(size_threshold_vox) <= 0:
        return labelled_image

    binary = labelled_image > 0
    binary = morphology.remove_small_objects(binary, min_size=int(size_threshold_vox), connectivity=1)
    return labelled_image * binary


def get_border_labels(labelled_image: np.ndarray) -> np.ndarray:
    """
    Returns labels that do NOT touch any border (excluding 0).
    """
    unique = np.unique(labelled_image)

    b0 = np.unique(labelled_image[0])
    b1 = np.unique(labelled_image[-1])
    b2 = np.unique(labelled_image[:, 0, :])
    b3 = np.unique(labelled_image[:, -1, :])
    b4 = np.unique(labelled_image[:, :, 0])
    b5 = np.unique(labelled_image[:, :, -1])

    border = np.unique(np.concatenate([b0, b1, b2, b3, b4, b5]))
    keep = np.array([i for i in unique if i not in border], dtype=unique.dtype)

    keep = keep[keep != 0]
    return keep


def delete_border_labels(labelled_image: np.ndarray) -> np.ndarray:
    """
    Zeros out any connected component label that touches the 3D volume border.
    """
    keep_labels = get_border_labels(labelled_image)
    mask = np.isin(labelled_image, keep_labels)

    cleaned = labelled_image.copy()
    cleaned[~mask] = 0
    return cleaned
