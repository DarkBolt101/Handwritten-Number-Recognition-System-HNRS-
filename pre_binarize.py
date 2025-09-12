import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Config ---
INPUT_TRAIN_CSV = "mnist_train.csv"
INPUT_TEST_CSV = "mnist_test.csv"
THRESHOLD = 128
TRAIN_OUT = "mnist_train_binarized.csv"
TEST_OUT = "mnist_test_binarized.csv"
SHOW_N = 8

# --- Step 1: Load data ---
train_df = pd.read_csv(INPUT_TRAIN_CSV, header=None)
test_df = pd.read_csv(INPUT_TEST_CSV, header=None)

# --- Step 2: Split labels and pixels ---
label_train = train_df.iloc[:, 0].values
pixel_train = train_df.iloc[:, 1:].values
label_test = test_df.iloc[:, 0].values
pixel_test = test_df.iloc[:, 1:].values

# --- Step 3: Binarize ---
pixel_train_bin = (pixel_train >= THRESHOLD).astype(np.uint8)
pixel_test_bin = (pixel_test >= THRESHOLD).astype(np.uint8)

# --- Step 4: Save new processed CSVs ---
train_out_df = pd.DataFrame(np.column_stack([label_train, pixel_train_bin]))
test_out_df = pd.DataFrame(np.column_stack([label_test, pixel_test_bin]))
train_out_df.to_csv(TRAIN_OUT, index=False, header=False)
test_out_df.to_csv(TEST_OUT,  index=False, header=False)

# --- Step 5: Visual check ---
imgs = pixel_train_bin[:SHOW_N].reshape(-1, 28, 28)
labels = label_train[:SHOW_N]

plt.figure(figsize=(12, 3))
for i in range(SHOW_N):
    plt.subplot(1, SHOW_N, i+1)
    plt.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
    plt.title(f"Label: {labels[i]}")
    plt.axis("off")
plt.suptitle(f"First {SHOW_N} binarized digits (threshold = {THRESHOLD})")
plt.show()

print("Binarization complete!")
print(f"Train dataset saved as: {TRAIN_OUT}")
print(f"Test dataset saved as: {TEST_OUT}")