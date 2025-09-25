import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops, regionprops_table
from visualization import Visualization

def segmentation(file_path, visualize=False, save_path=None):
    img = cv2.imread(file_path, 0)
    if img is None:
        raise ValueError(f"Failed to read image: {file_path}")

    image_label = label(img)
    regions = regionprops(image_label, intensity_image=img)

    props = regionprops_table(
        image_label, img,
        properties=['label', 'area', 'equivalent_diameter', 'mean_intensity', 'solidity']
    )
    df = pd.DataFrame(props)

    if len(df) == 0:
        mask = np.zeros_like(img, dtype=bool)
        convert_image = (mask.astype(np.uint8)) * 255
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(convert_image, cv2.MORPH_OPEN, kernel)
        tot_mean_intensity = 0.0
        average_mean_intensity = 0.0
    else:
        tot_mean_intensity = float(df['mean_intensity'].sum())
        average_mean_intensity = tot_mean_intensity / len(df)

        mask = np.zeros_like(img, dtype=bool)
        for region in regions:
            if region.intensity_mean >= average_mean_intensity + 10:
                mask[image_label == region.label] = True
        convert_image = (mask.astype(np.uint8)) * 255
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(convert_image, cv2.MORPH_OPEN, kernel)

    if visualize:
        Visualization.seg_visual(img, mask, opening, tot_mean_intensity, average_mean_intensity, save_path=save_path)

    return img, mask, opening, tot_mean_intensity, average_mean_intensity
