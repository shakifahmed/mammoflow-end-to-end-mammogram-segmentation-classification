import os
import cv2
import imutils
import glob

def augmentation(input_dir, output_dir, apply_flipping=True, apply_rotation=True, 
                            rotation_angles=None, custom_resize=(224, 224)):
 
    if rotation_angles is None:
        rotation_angles = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        # rotation_angles = [30, 60]
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    flip_types = [0, 1] if apply_flipping else []
    
    # Process all subdirectories
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    total_original = 0
    total_augmented = 0
    
    for subdir in subdirs:
        input_subdir = os.path.join(input_dir, subdir)
        output_subdir = os.path.join(output_dir, subdir)
        
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        
        image_paths = glob.glob(os.path.join(input_subdir, '*.png'))
        total_original += len(image_paths)
        
        print(f"Processing {subdir}: {len(image_paths)} images")
        
        for image_path in image_paths:
            filename = os.path.basename(image_path)
            base_name = os.path.splitext(filename)[0]
            
            img = cv2.imread(image_path, 0)
            if img is None:
                continue
            
            # Save original
            cv2.imwrite(os.path.join(output_subdir, filename), img)
            total_augmented += 1
            
            # Apply flipping
            if apply_flipping:
                for flip_type in flip_types:
                    flipped_img = cv2.flip(img, flip_type)
                    flip_suffix = "vflip" if flip_type == 0 else "hflip"
                    
                    counter = 1
                    augmented_filename = f"{base_name}_{flip_suffix}.png"
                    while os.path.exists(os.path.join(output_subdir, augmented_filename)):
                        augmented_filename = f"{base_name}_{flip_suffix}({counter}).png"
                        counter += 1
                    
                    cv2.imwrite(os.path.join(output_subdir, augmented_filename), flipped_img)
                    total_augmented += 1
            
            # Apply rotation
            if apply_rotation:
                for angle in rotation_angles:
                    rotated_img = imutils.rotate_bound(img, angle)
                    if custom_resize:
                        rotated_img = cv2.resize(rotated_img, custom_resize)
                    
                    counter = 1
                    augmented_filename = f"{base_name}_rot{angle}.png"
                    while os.path.exists(os.path.join(output_subdir, augmented_filename)):
                        augmented_filename = f"{base_name}_rot{angle}({counter}).png"
                        counter += 1
                    
                    cv2.imwrite(os.path.join(output_subdir, augmented_filename), rotated_img)
                    total_augmented += 1
    
    print(f"\nAugmentation Summary:")
    print(f"Original images: {total_original}")
    print(f"Total images after augmentation: {total_augmented}")
    # print(f"Augmentation factor: {total_augmented/total_original:.1f}x")