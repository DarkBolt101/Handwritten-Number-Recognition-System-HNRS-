"""
Digit Splitting Methods Module
"""

import numpy as np
import cv2
from skimage.morphology import skeletonize

# =============================================================================
# TRACKING
# =============================================================================

_last_splitting_method = "None"

def get_last_splitting_method():
    """Get last splitting method used."""
    return _last_splitting_method

def track_splitting_method(method_name):
    """Track splitting method usage."""
    global _last_splitting_method
    _last_splitting_method = method_name

# =============================================================================
# SPLITTING METHODS
# =============================================================================

def split_touching_digits(mask, contour, split_method="simple"):
    """Split touching digits using specified method."""
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 1
    
    # Only attempt splitting for wide contours
    if aspect_ratio < 1.25:
        return [contour]
    
    roi_mask = mask[y:y+h, x:x+w]
    
    # For very wide contours, try recursive splitting
    if aspect_ratio > 2.0:
        parts = _split_recursively(roi_mask, split_method, max_digits=6)
        if len(parts) > 1:
            track_splitting_method(f"{split_method} (recursive, {len(parts)} parts)")
            return _convert_parts_to_contours(parts, x, y)
    
    # Apply single splitting method
    if split_method == "simple":
        return _split_simple(roi_mask, x, y)
    elif split_method == "projection":
        return _split_projection(roi_mask, x, y)
    elif split_method == "kmeans1d":
        return _split_kmeans1d(roi_mask, x, y)
    elif split_method == "skeleton":
        return _split_skeleton(roi_mask, x, y)
    elif split_method == "auto":
        return _split_smart(roi_mask, x, y)
    else:
        track_splitting_method(f"Unknown method: {split_method}")
        return [contour]

def _split_simple(roi_mask, x, y):
    """Simple middle split method."""
    h, w = roi_mask.shape
    middle = w // 2
    left_mask = roi_mask[:, :middle]
    right_mask = roi_mask[:, middle:]
    
    if np.count_nonzero(left_mask) < 10 or np.count_nonzero(right_mask) < 10:
        track_splitting_method("simple (failed - parts too small)")
        return [np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)]
    
    left_contour = _mask_to_contour(left_mask, x, y)
    right_contour = _mask_to_contour(right_mask, x + middle, y)
    
    if left_contour is not None and right_contour is not None:
        track_splitting_method("simple (successful)")
        return [left_contour, right_contour]
    else:
        track_splitting_method("simple (failed - no valid contours)")
        return [np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)]

def _split_projection(roi_mask, x, y):
    """Projection-based splitting using valley detection."""
    h, w = roi_mask.shape
    if w < 20:
        track_splitting_method("projection (failed - too narrow)")
        return [np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)]
    
    # Get column projection
    col_sum = np.sum(roi_mask > 0, axis=0).astype(np.float32)
    if col_sum.max() == 0:
        track_splitting_method("projection (failed - no foreground)")
        return [np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)]
    
    # Smooth the projection
    k = max(3, w // 20)
    kernel = np.ones(k, dtype=np.float32) / k
    smooth = np.convolve(col_sum, kernel, mode='same')
    
    # Find valleys (local minima)
    valleys = []
    for i in range(2, w - 2):
        if (smooth[i] < smooth[i-1] and 
            smooth[i] < smooth[i+1] and 
            smooth[i] < smooth.max() * 0.3):  # Significant valley
            valleys.append(i)
    
    if not valleys:
        track_splitting_method("projection (failed - no valleys found)")
        return [np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)]
    
    # Split at the deepest valley
    split_x = valleys[np.argmin([smooth[v] for v in valleys])]
    left = roi_mask[:, :split_x]
    right = roi_mask[:, split_x:]
    
    if np.count_nonzero(left) < 10 or np.count_nonzero(right) < 10:
        track_splitting_method("projection (failed - parts too small)")
        return [np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)]
    
    track_splitting_method("projection (successful)")
    return _convert_parts_to_contours([left, right], x, y)

def _split_kmeans1d(roi_mask, x, y):
    """1D k-means splitting on x-coordinates."""
    h, w = roi_mask.shape
    if w < 20:
        track_splitting_method("kmeans1d (failed - too narrow)")
        return [np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)]
    
    # Get x-coordinates of foreground pixels
    y_coords, x_coords = np.where(roi_mask > 0)
    if len(x_coords) < 20:
        track_splitting_method("kmeans1d (failed - not enough pixels)")
        return [np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)]
    
    # 1D k-means on x-coordinates
    data = x_coords.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Split at the mean of cluster centers
    split_x = int(np.mean(centers))
    left = roi_mask[:, :split_x]
    right = roi_mask[:, split_x:]
    
    if np.count_nonzero(left) < 10 or np.count_nonzero(right) < 10:
        track_splitting_method("kmeans1d (failed - parts too small)")
        return [np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)]
    
    track_splitting_method("kmeans1d (successful)")
    return _convert_parts_to_contours([left, right], x, y)

def _split_skeleton(roi_mask, x, y):
    """Skeleton-based splitting."""
    h, w = roi_mask.shape
    if w < 20:
        track_splitting_method("skeleton (failed - too narrow)")
        return [np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)]
    
    # Create skeleton
    binary = (roi_mask > 0).astype(np.uint8)
    skeleton = skeletonize(binary)
    
    # Get column projection of skeleton
    col_sum = np.sum(skeleton > 0, axis=0)
    if col_sum.size < 3:
        track_splitting_method("skeleton (failed - no skeleton)")
        return [np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)]
    
    # Find minimum column
    min_val = np.min(col_sum)
    avg_val = np.mean(col_sum)
    
    # Only split if there's a clear valley
    if min_val > avg_val * 0.3:
        track_splitting_method("skeleton (failed - no clear valley)")
        return [np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)]
    
    split_x = int(np.argmin(col_sum))
    
    # Don't split too close to edges
    if split_x <= 2 or split_x >= w - 2:
        track_splitting_method("skeleton (failed - too close to edges)")
        return [np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)]
    
    left = roi_mask[:, :split_x]
    right = roi_mask[:, split_x:]
    
    if np.count_nonzero(left) < 15 or np.count_nonzero(right) < 15:
        track_splitting_method("skeleton (failed - parts too small)")
        return [np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)]
    
    track_splitting_method("skeleton (successful)")
    return _convert_parts_to_contours([left, right], x, y)

def _split_smart(roi_mask, x, y):
    """Smart splitting: try simple first, then advanced methods."""
    h, w = roi_mask.shape
    
    # First try simple split
    middle = w // 2
    left_mask = roi_mask[:, :middle]
    right_mask = roi_mask[:, middle:]
    
    if np.count_nonzero(left_mask) >= 10 and np.count_nonzero(right_mask) >= 10:
        track_splitting_method("simple (successful)")
        return _convert_parts_to_contours([left_mask, right_mask], x, y)
    
    # If simple fails, try advanced methods
    methods = [
        (_split_projection, "projection"),
        (_split_kmeans1d, "kmeans1d"),
        (_split_skeleton, "skeleton")
    ]
    
    for method_func, method_name in methods:
        result = method_func(roi_mask, x, y)
        if len(result) == 2:
            track_splitting_method(f"{method_name} (successful)")
            return result
    
    track_splitting_method("all methods failed")
    return [np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)]

def _split_recursively(roi_mask, split_method, max_digits=6):
    """Recursively split wide regions into multiple parts."""
    parts = [roi_mask]
    
    for _ in range(max_digits - 1):
        new_parts = []
        for part in parts:
            h, w = part.shape
            aspect_ratio = w / h if h > 0 else 1
            
            # Only split if still wide enough
            if aspect_ratio > 1.2:
                if split_method == "simple":
                    middle = w // 2
                    left = part[:, :middle]
                    right = part[:, middle:]
                    if np.count_nonzero(left) >= 10 and np.count_nonzero(right) >= 10:
                        new_parts.extend([left, right])
                    else:
                        new_parts.append(part)
                elif split_method == "projection":
                    split_parts = _split_projection(part, 0, 0)
                    if len(split_parts) == 2:
                        # Extract masks from contours
                        left_mask = _contour_to_mask(split_parts[0], part.shape)
                        right_mask = _contour_to_mask(split_parts[1], part.shape)
                        new_parts.extend([left_mask, right_mask])
                    else:
                        new_parts.append(part)
                elif split_method == "kmeans1d":
                    split_parts = _split_kmeans1d(part, 0, 0)
                    if len(split_parts) == 2:
                        left_mask = _contour_to_mask(split_parts[0], part.shape)
                        right_mask = _contour_to_mask(split_parts[1], part.shape)
                        new_parts.extend([left_mask, right_mask])
                    else:
                        new_parts.append(part)
                elif split_method == "skeleton":
                    split_parts = _split_skeleton(part, 0, 0)
                    if len(split_parts) == 2:
                        left_mask = _contour_to_mask(split_parts[0], part.shape)
                        right_mask = _contour_to_mask(split_parts[1], part.shape)
                        new_parts.extend([left_mask, right_mask])
                    else:
                        new_parts.append(part)
                else:
                    new_parts.append(part)
            else:
                new_parts.append(part)
        
        # If no more splits possible, stop
        if len(new_parts) == len(parts):
            break
        parts = new_parts
    
    return parts

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _mask_to_contour(mask, offset_x, offset_y):
    """Convert mask to contour with offset."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Get largest contour
    largest = max(contours, key=cv2.contourArea)
    # Add offset
    return largest + np.array([offset_x, offset_y])

def _contour_to_mask(contour, shape):
    """Convert contour back to mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    return mask

def _convert_parts_to_contours(parts, x, y):
    """Convert split parts back to contours."""
    contours = []
    current_x = x
    
    for part in parts:
        if part is None or part.size == 0:
            continue
            
        part_h, part_w = part.shape
        if part_w < 5 or part_h < 5:
            continue
            
        # Find contours in this part
        part_contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if part_contours:
            # Get largest contour and add offset
            largest = max(part_contours, key=cv2.contourArea)
            contours.append(largest + np.array([current_x, y]))
        
        current_x += part_w
    
    # If no contours found, return a simple rectangle
    if not contours:
        w = sum(part.shape[1] for part in parts if part is not None and part.size > 0)
        h = max(part.shape[0] for part in parts if part is not None and part.size > 0) if parts else 20
        return [np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)]
    
    return contours
