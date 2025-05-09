# prepare_training_data.py
import os
import cv2
import numpy as np

def preprocess_training_data(style_dir, output_size=(256, 256)):
    """Process all images in a style directory to consistent format"""
    processed_images = []
    for filename in os.listdir(style_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(style_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, output_size)
            processed_images.append(img)
    
    # Save as numpy array for faster loading
    if processed_images:
        np.save(os.path.join(style_dir, "processed_data.npy"), 
                np.array(processed_images))
        print(f"Processed {len(processed_images)} images for {os.path.basename(style_dir)}")

if __name__ == "__main__":
    # Process each style folder
    for style in ['cyberpunk', 'industrial', 'nature', 'abstract']:
        style_dir = f"training/styles/{style}"
        if os.path.exists(style_dir):
            preprocess_training_data(style_dir)
        else:
            print(f"Directory not found: {style_dir}")