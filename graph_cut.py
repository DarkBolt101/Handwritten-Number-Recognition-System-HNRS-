import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Input and output folders
input_folder = "images/"
output_folder = "segmented_output/graphcut/"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

segmented_images = []
filenames = []

# Loop through all PNG files in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Convert to RGB for visualization later
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Initialize mask and models for GrabCut
        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # Define rectangle covering the object (x, y, w, h)
        # For MNIST, digits are centered, so rectangle slightly smaller than full image
        h, w = image.shape[:2]
        rect = (1, 1, w-2, h-2)  # avoid border pixels

        # Apply GrabCut
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        # Prepare final mask: probable/definite foreground
        final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # Apply mask to original image
        segmented_image = image_rgb * final_mask[:, :, np.newaxis]

        # Save segmented image
        output_path = os.path.join(output_folder, f"graphcut_{filename}")
        cv2.imwrite(output_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

        segmented_images.append(segmented_image)
        filenames.append(filename)

# Display all segmented images in a grid
n = len(segmented_images)
cols = 5
rows = (n + cols - 1) // cols

plt.figure(figsize=(15, 3 * rows))
for i, (img, fname) in enumerate(zip(segmented_images, filenames)):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(img)
    plt.title(fname, fontsize=8)
    plt.axis("off")

plt.tight_layout()
plt.show()
print(f"GraphCut segmentation complete. Results saved in '{output_folder}'")
