"""
Digit Extraction Module
"""

import numpy as np
import cv2
from segmentation import segment_image_combined
from splitting import split_touching_digits

# =============================================================================
# TRACKING
# =============================================================================

_last_splitting_method = "None"
_last_segmentation_method = "Unknown"

def get_last_splitting_method():
    """Get last splitting method used."""
    return _last_splitting_method

def get_last_segmentation_method():
    """Get last segmentation method used."""
    return _last_segmentation_method

def track_splitting_method(method_name):
    """Track splitting method usage."""
    global _last_splitting_method
    _last_splitting_method = method_name

def track_segmentation_method(method_name):
    """Track segmentation method usage."""
    global _last_segmentation_method
    _last_segmentation_method = method_name

# =============================================================================
# MNIST CHIP CREATION
# =============================================================================

def create_mnist_chip(roi_mask, roi_gray, use_grayscale=True):
    """Create MNIST-style 28x28 chip."""
    h, w = roi_mask.shape
    
    if use_grayscale and roi_gray is not None and roi_gray.size > 0:
        # Use grayscale with centering
        chip = roi_gray.astype(np.float32) / 255.0
    else:
        # Use binary mask
        chip = (roi_mask > 0).astype(np.float32)
    
    # Pad to square
    max_dim = max(h, w)
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2
    chip = np.pad(chip, ((pad_h, max_dim - h - pad_h), (pad_w, max_dim - w - pad_w)), mode='constant')
    
    # Resize to 20x20
    chip = cv2.resize(chip, (20, 20), interpolation=cv2.INTER_AREA)
    
    # Pad to 28x28
    chip = np.pad(chip, 4, mode='constant')
    
    # Center using moments
    moments = cv2.moments(chip)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        
        # Create transformation matrix
        M = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
        chip = cv2.warpAffine(chip, M, (28, 28))
    
    # Ensure white digit on black background
    if np.mean(chip) > 0.5:
        chip = 1.0 - chip
    
    return chip

def create_chips_display(chips):
    """Create a display image for the digit chips."""
    if not chips:
        return None
    
    # Create a grid layout for chips
    cols = min(5, len(chips))
    rows = (len(chips) + cols - 1) // cols
    
    chip_size = 28
    spacing = 5
    width = cols * chip_size + (cols - 1) * spacing
    height = rows * chip_size + (rows - 1) * spacing
    
    display_img = np.ones((height, width), dtype=np.uint8) * 255
    
    for i, chip in enumerate(chips):
        row = i // cols
        col = i % cols
        
        x = col * (chip_size + spacing)
        y = row * (chip_size + spacing)
        
        # Convert chip to 0-255 range
        chip_255 = (chip * 255).astype(np.uint8)
        display_img[y:y+chip_size, x:x+chip_size] = chip_255
    
    return display_img

# =============================================================================
# MAIN EXTRACTION FUNCTION
# =============================================================================

def extract_digits_combined(mask, gray_image, 
                           segmentation_method="auto",
                           split_method="simple",
                           use_grayscale_chips=True,
                           enable_pre_erosion=False,
                           one_digit_only=False):
    """
    Extract digits using combined methods.
    
    Args:
        mask: Binary mask from segmentation
        gray_image: Original grayscale image
        segmentation_method: Segmentation method used
        split_method: Splitting method ("simple", "projection", "kmeans1d", "skeleton", "smart")
        use_grayscale_chips: Use grayscale for chips or binary
        enable_pre_erosion: Apply erosion before splitting
        one_digit_only: Only extract one digit
    
    Returns:
        Tuple of (digit_chips, visualization_image)
    """
    # Apply pre-erosion if enabled
    if enable_pre_erosion:
        mask = cv2.erode(mask, np.ones((2, 2), np.uint8))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    
    if one_digit_only:
        contours = [max(contours, key=cv2.contourArea)]
    
    # Apply splitting for touching digits
    all_contours = []
    splitting_occurred = False
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h > 0 else 1
        
        # Split wide contours that might contain multiple digits
        if aspect_ratio >= 1.25:
            split_contours = split_touching_digits(mask, cnt, split_method)
            all_contours.extend(split_contours)
            if len(split_contours) > 1:
                splitting_occurred = True
        else:
            all_contours.append(cnt)
    
    # Track splitting method used
    if len(all_contours) > 1:
        if not splitting_occurred:
            track_splitting_method(f"Natural separation ({len(all_contours)} digits)")
    else:
        track_splitting_method("None (single digit)")
    
    # Track segmentation method
    track_segmentation_method(segmentation_method)
    
    vis = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    chips, boxes = [], []
    
    for cnt in all_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Skip tiny regions
        if w * h < 50:
            continue
        
        # Extract ROI
        roi_mask = mask[y:y+h, x:x+w]
        roi_gray = gray_image[y:y+h, x:x+w]
        
        # Create MNIST-style chip
        chip = create_mnist_chip(roi_mask, roi_gray, use_grayscale_chips)
        if chip is None:
            continue
        
        # Validate chip quality
        fg_ratio = np.sum(chip > 0.1) / (28 * 28)
        if fg_ratio < 0.01:
            continue
        
        chips.append(chip)
        
        # Calculate tight bounding box for visualization
        ys, xs = np.where(roi_mask > 0)
        if xs.size > 0 and ys.size > 0:
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            boxes.append((x + x0, y + y0, x1 - x0 + 1, y1 - y0 + 1))
    
    # Sort chips left-to-right
    if boxes:
        order = np.argsort([bx for bx, _, _, _ in boxes])
        chips = [chips[i] for i in order]
        boxes = [boxes[i] for i in order]
        
        # Redraw visualization with sorted order
        new_vis = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        for bx, by, bw, bh in boxes:
            cv2.rectangle(new_vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        vis = new_vis
    
    return chips, vis
