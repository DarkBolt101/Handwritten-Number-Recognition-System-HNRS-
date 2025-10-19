"""
Segmentation Methods Module
"""

import numpy as np
import cv2

# =============================================================================
# SEGMENTATION METHODS
# =============================================================================

def segment_image_combined(gray_image, method="auto"):
    """Segment image using various methods."""
    # Track method used
    from digit_extraction import track_segmentation_method
    if method == "auto":
        track_segmentation_method("Otsu")
    else:
        track_segmentation_method(method.upper())
    
    if method in ("auto", "otsu"):
        return _segment_otsu(gray_image)
    elif method == "adaptive":
        return _segment_adaptive(gray_image)
    elif method == "kmeans":
        return _segment_kmeans(gray_image)
    elif method == "local_threshold":
        return _segment_local_threshold(gray_image)
    elif method == "canny_edges":
        return _segment_canny_edges(gray_image)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")

def _segment_otsu(gray_image):
    """Otsu thresholding segmentation."""
    blur = cv2.GaussianBlur(gray_image, (3, 3), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return _ensure_white_foreground(mask, gray_image)

def _segment_adaptive(gray_image):
    """Adaptive thresholding segmentation."""
    blur = cv2.GaussianBlur(gray_image, (3, 3), 0)
    mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return _ensure_white_foreground(mask, gray_image)

def _segment_kmeans(gray_image):
    """K-means clustering segmentation."""
    blur = cv2.GaussianBlur(gray_image, (3, 3), 0)
    data = blur.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    mask = labels.reshape(gray_image.shape).astype(np.uint8) * 255
    return _ensure_white_foreground(mask, gray_image)

def _segment_local_threshold(gray_image):
    """Local adaptive thresholding segmentation."""
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    return _ensure_white_foreground(mask, gray_image)

def _segment_canny_edges(gray_image):
    """Canny edge-based segmentation."""
    blur = cv2.GaussianBlur(gray_image, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)
    return _ensure_white_foreground(mask, gray_image)

def _ensure_white_foreground(mask, gray_image):
    """Ensure white digits on black background."""
    if np.mean(mask[gray_image > 128]) < np.mean(mask[gray_image <= 128]):
        mask = 255 - mask
    return mask
