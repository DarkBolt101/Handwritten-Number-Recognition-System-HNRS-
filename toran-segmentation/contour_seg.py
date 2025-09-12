import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def contourSeg():

    input_folder = "images/"
    output_folder = "segmented_output/"

    os.makedirs(output_folder, exist_ok=True)

    segmented_images = []
    filenames = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue  

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Thresholding
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            segmented_image = image.copy()

            for contour in contours:
                # Approximate the contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx_polygon = cv2.approxPolyDP(contour, epsilon, True)

                # Draw the approximated polygon
                cv2.drawContours(segmented_image, [approx_polygon], -1, (0, 255, 0), 1)  # Green polygon

            output_path = os.path.join(output_folder, f"segmented_{filename}")
            cv2.imwrite(output_path, segmented_image)

            # Convert BGR to RGB for Matplotlib
            segmented_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

            segmented_images.append(segmented_rgb)
            filenames.append(filename)

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

    print(f"Segmentation complete. Results saved in '{output_folder}'")

if __name__ == '__main__':
    contourSeg()
