import os
import cv2
import random
from glob import glob
import albumentations as A

# Augmentation pipeline
augmentor = A.Compose([
    A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.HueSaturationValue(p=0.3),
    A.RGBShift(p=0.2),
    A.MotionBlur(p=0.1),
    A.GaussNoise(p=0.1),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
])

def augment_images(input_folder, output_folder, total_augmented_images):
    os.makedirs(output_folder, exist_ok=True)
    image_paths = glob(os.path.join(input_folder, "*.jpg")) + \
                  glob(os.path.join(input_folder, "*.jpeg")) + \
                  glob(os.path.join(input_folder, "*.png"))
    
    if not image_paths:
        print("No images found.")
        return

    current_count = 0
    while current_count < total_augmented_images:
        img_path = random.choice(image_paths)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        augmented = augmentor(image=img)['image']
        aug_filename = f"aug_{current_count}.jpg"
        cv2.imwrite(os.path.join(output_folder, aug_filename), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
        current_count += 1

    print(f"Saved {current_count} augmented images to {output_folder}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IN_PATH = os.path.join(BASE_DIR, "data", "raw", "headphones")   #replace with your actual path
OUT_PATH = os.path.join(BASE_DIR, "data", "raw", "headphones")  #replace with your actual path
num = 1000  # Number of augmented images to generate
# Example usage:
augment_images(IN_PATH, OUT_PATH, num)