import os
import re
import glob
import gc
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from skimage import io
import tifffile


def upload_images(image_path: str) -> np.ndarray:
    """
    Load 3D image data from:
      - a folder containing multiple 2D .tif/.tiff slices, OR
      - a folder containing a single 3D .tif/.tiff

    Returns:
      images: (Z,Y,X) ndarray
    """
    image_path = os.path.normpath(image_path) + os.sep

    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', os.path.basename(s))]

    tiff_files = sorted(
        glob.glob(image_path + "*.tif") + glob.glob(image_path + "*.tiff"),
        key=natural_sort_key
    )

    if len(tiff_files) > 1:
        def process_tiff(img_path):
            return io.imread(img_path)

        cv_img = Parallel(n_jobs=-1)(
            delayed(process_tiff)(img) for img in tqdm(tiff_files, desc="Uploading TIFF slices")
        )
        images = np.stack(cv_img, axis=0)  # (Z,Y,X)
        del cv_img
        gc.collect()

    elif len(tiff_files) == 1:
        tiff_file = tiff_files[0]
        with tifffile.TiffFile(tiff_file) as tif:
            num_slices = len(tif.pages)

        def process_slice(i):
            with tifffile.TiffFile(tiff_file) as tif:
                return tif.pages[i].asarray()

        images = Parallel(n_jobs=-1)(
            delayed(process_slice)(i) for i in tqdm(range(num_slices), desc="Uploading 3D TIFF pages")
        )
        images = np.asarray(images)
        gc.collect()

    else:
        raise ValueError("No .tif/.tiff files found in the specified folder.")

    # Standardize dtype (optional)
    max_grey_value = np.max(images)
    if max_grey_value <= 255:
        images = images.astype(np.uint8, copy=False)
    elif max_grey_value <= 65535:
        images = images.astype(np.uint16, copy=False)
    else:
        images = images.astype(np.uint32, copy=False)

    gc.collect()
    return images


def save_images(vol: np.ndarray, saving_path: str, folder_name: str, gc_every: int = 64) -> None:
    """
    Save a 3D volume (Z,Y,X) as individual 2D TIFF slices.

    Creates:
      saving_path / folder_name / 0000.tif ...

    - If input is bool: saved as uint8 with 0 and 255 (viewer-friendly)
    - Otherwise: saved as-is per-slice
    """
    output_dir = os.path.join(saving_path, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    Z = vol.shape[0]

    for z in tqdm(range(Z), desc=f"Saving {folder_name}"):
        slice2d = vol[z]

        if slice2d.dtype == bool:
            slice2d = slice2d.astype(np.uint8) * 255
        else:
            slice2d = np.asarray(slice2d)

        tifffile.imwrite(os.path.join(output_dir, f"{z:04d}.tif"), slice2d)

        if gc_every and (z % gc_every) == 0:
            gc.collect()

    gc.collect()
