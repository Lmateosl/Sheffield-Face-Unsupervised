"""
Extract random sample images from UMIST dataset for testing.
Creates one sample image per person, saved with appropriate filenames.
"""
import os
import scipy.io
import numpy as np
from PIL import Image

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory for sample images
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sample_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the UMIST dataset
print("Loading UMIST dataset...")
mat = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), 'umist_cropped.mat'))
facedat = mat['facedat']

print(f"Dataset contains {facedat.shape[1]} persons")
print("\nExtracting random sample images...")

# Extract one random image per person
for person_idx in range(facedat.shape[1]):
    # Get all images for this person (shape: 112 x 92 x num_images)
    person_images = facedat[0, person_idx]
    num_images = person_images.shape[2]
    
    # Randomly select one image
    random_img_idx = np.random.randint(0, num_images)
    selected_image = person_images[:, :, random_img_idx]
    
    # Normalize to 0-255 range for proper image display
    # The dataset values might be in different ranges
    img_normalized = selected_image.astype(np.float64)
    img_normalized = (img_normalized - img_normalized.min()) / (img_normalized.max() - img_normalized.min())
    img_normalized = (img_normalized * 255).astype(np.uint8)
    
    # Convert to PIL Image and save
    img = Image.fromarray(img_normalized, mode='L')  # 'L' mode for grayscale
    
    # Save with person number (using 0-based indexing like in the model)
    filename = f"person_{person_idx}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    img.save(filepath)
    
    print(f"  Saved person {person_idx}: {filename} (selected image {random_img_idx+1}/{num_images})")

print(f"\n✓ Successfully saved {facedat.shape[1]} sample images to: {OUTPUT_DIR}")
print("\nYou can now drag and drop these images to test your frontend interface!")
print(f"Files are named: person_0.png, person_1.png, ..., person_{facedat.shape[1]-1}.png")
