import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

#################################
# Configuration #
#################################
IMAGE_PATH = 'numbers.png'
METHOD = 'canny_edges' # Each Method: 'threshold','kmeans','local_threshold','canny_edges'
SHOW_BBOXES = True # Whether to draw bounding boxes around detected digits
MIN_COMPONENT_AREA = 30 # The minimum pixel area for a component to be considered a digit
ENABLE_TOUCHING_SPLIT = True # Whether to try splitting connected digits

# Mapping segmentation methods to splitting strategies
METHOD_TO_SPLIT = {
    'threshold': 'smart_split',  # Combination of projection/kmeans/skeleton
    'kmeans': 'smart_split',
    'local_threshold': 'projection',  # Use vertical projection for adaptive threshold
    'canny_edges': 'skeleton'  # Use skeletonization for Canny edges
}
SPLIT_STRATEGY = METHOD_TO_SPLIT.get(METHOD, 'smart_split')  # Default strategy

#################################
# Utility functions #
#################################
def show_steps(images, titles, main_title):
    """ Display multiple images side by side for debugging/visualization """
    cols = len(images)
    plt.figure(figsize=(4*cols, 4))
    for i, (img, title) in enumerate(zip(images, titles), 1):
        cmap = 'gray' if len(img.shape) == 2 else None  # Use gray colormap for 2D images
        plt.subplot(1, cols, i)
        plt.imshow(img, cmap=cmap)
        plt.title(f"{i}. {title}")
        plt.axis('off')
    plt.suptitle(main_title, fontsize=14)
    plt.tight_layout()
    plt.show()

def pad_to_square(img, pad_value=0):
    """ Pad rectangular image to square shape for consistent resizing """
    h, w = img.shape[:2]
    if h == w: return img
    if h > w:
        pad = h - w; left = pad // 2; right = pad - left
        return np.pad(img, ((0,0), (left,right)), mode='constant', constant_values=pad_value)
    else:
        pad = w - h; top = pad // 2; bottom = pad - top
        return np.pad(img, ((top,bottom), (0,0)), mode='constant', constant_values=pad_value)

#################################
# Image Segmentation Methods #
##################################
def segment_threshold(gray):
    """ Segment image using Gaussian blur + Otsu's thresholding """
    blur = cv2.GaussianBlur(gray, (5,5), 0)  # Smooth to reduce noise
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Inverted threshold
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)  # Remove small noise
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)  # Fill small holes
    show_steps([gray, blur, th, cleaned], ['Original','Blurred','Otsu Threshold','Cleaned'], 'Threshold Segmentation')
    return cleaned

def segment_kmeans(gray):
    """ Segment image using k-means clustering """
    blur = cv2.GaussianBlur(gray, (5,5), 0)  # Smooth before clustering
    pixels = blur.reshape((-1,1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    clustered = centers[labels.flatten()].reshape(gray.shape)
    # Treat the darker cluster as foreground
    fg = (clustered == min(centers.flatten())).astype(np.uint8)*255
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)  # Remove small noise
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)  # Fill holes
    show_steps([gray, blur, clustered, fg, cleaned], ['Original','Blurred','Clustered','Binary','Cleaned'], 'K-means Segmentation')
    return cleaned

def segment_local_threshold(gray):
    """ Segment using adaptive local thresholding """
    blur = cv2.GaussianBlur(gray, (5,5), 0)  # Reduce noise
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 10)
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    show_steps([gray, blur, adaptive, cleaned], ['Original','Blurred','Adaptive','Cleaned'], 'Local Threshold')
    return cleaned

def segment_canny_edges(gray):
    """ Segment using Canny edge detection and fill detected edges """
    blur = cv2.GaussianBlur(gray, (5,5),0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel)  # Connect broken edges
    filled = np.zeros_like(gray)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)  # Fill digit regions
    cleaned = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)  # Smooth mask
    show_steps([gray, blur, edges, dilated, filled, cleaned], ['Original','Blur','Edges','Dilated','Filled','Cleaned'], 'Canny Edge Segmentation')
    return cleaned

#################################
# Splitting functions #
#################################
def split_by_projection(roi):
    """ Split connected digits using vertical projection profile """
    h, w = roi.shape
    if w < 10: return [roi]  # Too narrow to split
    col_sum = np.sum(roi>0, axis=0)
    k = max(3, w//20 | 1)
    smooth = np.convolve(col_sum, np.ones(k)/k, mode='same')  # Smooth projection
    margin = max(2, w//12)
    search = smooth[margin:w-margin]
    if search.size==0: return [roi]
    split_x = int(np.argmin(search)+margin)
    left, right = roi[:, :split_x], roi[:, split_x:]
    if np.count_nonzero(left)<10 or np.count_nonzero(right)<10: return [roi]  # Skip tiny regions
    return [left,right]

def split_by_kmeans1d(roi):
    """ Split connected digits using 1D k-means on x-coordinates """
    ys, xs = np.where(roi>0)
    if xs.size<20: return [roi]  # Not enough pixels to split
    xs = xs.astype(np.float32).reshape(-1,1)
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,50,0.5)
    _, _, centers = cv2.kmeans(xs, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    cut = int(centers.mean())
    left,right = roi[:, :cut], roi[:, cut:]
    if np.count_nonzero(left)<10 or np.count_nonzero(right)<10: return [roi]
    return [left,right]

def split_by_skeleton(roi):
    """ Split using skeletonization: cut at column with minimal skeleton pixels """
    skeleton = skeletonize((roi>0).astype(np.uint8)).astype(np.uint8)*255
    col_sum = np.sum(skeleton>0, axis=0)
    if col_sum.size<3: return [roi]
    split_x = int(np.argmin(col_sum))
    if col_sum[split_x]==0 or split_x<=1 or split_x>=roi.shape[1]-2: return [roi]
    left,right = roi[:, :split_x], roi[:, split_x:]
    if np.count_nonzero(left)<10 or np.count_nonzero(right)<10: return [roi]
    return [left,right]

def split_touching_component(roi, strategy=SPLIT_STRATEGY):
    """ Decide if a connected component should be split, choose method based on strategy """
    h,w = roi.shape
    area = np.count_nonzero(roi)
    aspect = w/max(1,h)
    # Skip small or non-digit-like regions
    if area<MIN_COMPONENT_AREA or w<12 or aspect<0.8: return [roi]

    # Split according to chosen strategy
    if strategy=='projection': return split_by_projection(roi)
    elif strategy=='kmeans1d': return split_by_kmeans1d(roi)
    elif strategy=='skeleton': return split_by_skeleton(roi)
    elif strategy=='smart_split':
        parts = split_by_projection(roi)
        if len(parts)==1: parts = split_by_kmeans1d(roi)
        if len(parts)==1: parts = split_by_skeleton(roi)
        return parts
    return [roi]

#################################
# Digit Extraction #
#################################
def extract_and_normalize_digits(binary_mask, gray_img, show_bboxes=True):
    """ Extract individual digits and normalize them to 28x28 MNIST-style images """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)  # Convert to color for drawing boxes
    digits, boxes = [], []

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h < MIN_COMPONENT_AREA: continue  # Ignore tiny noise
        roi = binary_mask[y:y+h, x:x+w]
        sub_rois = [roi]

        # Split touching digits if enabled
        if ENABLE_TOUCHING_SPLIT:
            sub_rois = split_touching_component(roi)

        for sub in sub_rois:
            if np.count_nonzero(sub)<MIN_COMPONENT_AREA: continue
            ys, xs = np.where(sub>0)
            x0,x1 = xs.min(), xs.max()
            y0,y1 = ys.min(), ys.max()
            sub_crop = sub[y0:y1+1, x0:x1+1]

            # Normalize: pad → resize → final padding to 28x28
            sub_crop = pad_to_square(sub_crop)
            resized = cv2.resize(sub_crop,(20,20),interpolation=cv2.INTER_AREA)
            padded = np.pad(resized, ((4,4),(4,4)), mode='constant', constant_values=0)
            digits.append(padded.astype('float32')/255.0)
            boxes.append((x+x0, y+y0, x1-x0+1, y1-y0+1))

    # Sort digits left-to-right
    if boxes:
        order = np.argsort([bx for bx,_,_,_ in boxes])
        digits = [digits[i] for i in order]
        if show_bboxes:
            for i in order:
                x,y,w,h = boxes[i]
                cv2.rectangle(bbox_img,(x,y),(x+w,y+h),(0,255,0),2)  # Draw bounding boxes

    return digits, bbox_img

def main():
    # Load grayscale image
    gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if gray is None: raise FileNotFoundError(f"Cannot load image {IMAGE_PATH}")

    # Segment according to chosen method
    if METHOD=='threshold': binary = segment_threshold(gray)
    elif METHOD=='kmeans': binary = segment_kmeans(gray)
    elif METHOD=='local_threshold': binary = segment_local_threshold(gray)
    elif METHOD=='canny_edges': binary = segment_canny_edges(gray)
    else: raise ValueError("METHOD must be 'threshold','kmeans','local_threshold','canny_edges'")

    # Optional erosion to clean thin connections
    eroded = cv2.erode(binary, np.ones((2,2),np.uint8))
    show_steps([binary, eroded], ['Mask before erosion','Mask after erosion'], 'Pre-splitting touch-up')

    # Extract individual digit images
    digits, bbox_img = extract_and_normalize_digits(eroded, gray, SHOW_BBOXES)

    # Visualize results
    show_steps([gray, eroded, bbox_img], ['Original','Mask','Detected Digits'], 'Digit Extraction Overview')

    # Display all extracted digits in a grid
    if digits:
        cols = min(8,len(digits))
        rows = int(np.ceil(len(digits)/cols))
        plt.figure(figsize=(2.5*cols,2.5*rows))
        for i,d in enumerate(digits,1):
            plt.subplot(rows,cols,i)
            plt.imshow(d,cmap='gray',vmin=0,vmax=1)
            plt.title(f'Digit {i}')
            plt.axis('off')
        plt.suptitle(f'Extracted digits (count={len(digits)})',fontsize=14)
        plt.tight_layout()
        plt.show()

    print(f"Total digits ready for model input: {len(digits)}")

if __name__=='__main__':
    main()