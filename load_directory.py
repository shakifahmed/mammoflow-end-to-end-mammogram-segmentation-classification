import os
import cv2
from segmentation import segmentation

def directory(source_dir, output_dir, save_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for folder in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder)

        if os.path.isdir(folder_path):
            duplicate_folder_path = os.path.join(output_dir, folder)
            if not os.path.exists(duplicate_folder_path):
                os.makedirs(duplicate_folder_path)

            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)

                # Only process PNG files that are regular files
                if os.path.isfile(file_path) and filename.lower().endswith('.png'):
                    _, _, opening, _, _ = segmentation(file_path, visualize=False)
                    cv2.imwrite(os.path.join(duplicate_folder_path, filename), opening)
