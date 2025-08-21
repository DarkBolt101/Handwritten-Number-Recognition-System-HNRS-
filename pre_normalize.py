import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Config ---
INPUT_TRAIN_CSV = "mnist_train.csv"
INPUT_TEST_CSV = "mnist_test.csv"
TRAIN_OUT = "mnist_train_normalized.csv"
TEST_OUT = "mnist_test_normalized.csv"
SHOW_N = 8

# --- Step 1: Load data ---
train_df = pd.read_csv(INPUT_TRAIN_CSV, header=None)
test_df  = pd.read_csv(INPUT_TEST_CSV, header=None)

# --- Step 2: Split labels and pixels ---
label_train = train_df.iloc[:, 0].values
pixel_train = train_df.iloc[:, 1:].values
label_test = test_df.iloc[:, 0].values
pixel_test = test_df.iloc[:, 1:].values

# --- Step 3: Normalize ---
# Convert 0-255 → 0.0-1.0 (float32)
pixel_train_norm = (pixel_train / 255.0).astype(np.float32)
pixel_test_norm = (pixel_test / 255.0).astype(np.float32)

# --- Step 4: Save new processed CSVs ---
train_out_df = pd.DataFrame(np.column_stack([label_train, pixel_train_norm]))
test_out_df = pd.DataFrame(np.column_stack([label_test, pixel_test_norm]))
train_out_df.to_csv(TRAIN_OUT, index=False, header=False)
test_out_df.to_csv(TEST_OUT,  index=False, header=False)

print("Normalization complete!")
print(f"Train dataset saved as: {TRAIN_OUT}")
print(f"Test dataset saved as: {TEST_OUT}")
