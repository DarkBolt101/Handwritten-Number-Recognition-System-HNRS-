import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def normalize_mnist_to_csv(train_out="mnist_train_normalized.csv", test_out="mnist_test_normalized.csv", show_n=0):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train_norm = (x_train / 255.0).astype(np.float32)
    x_test_norm = (x_test / 255.0).astype(np.float32)
    
    x_train_flat = x_train_norm.reshape(x_train_norm.shape[0], -1)
    x_test_flat = x_test_norm.reshape(x_test_norm.shape[0], -1)
    
    train_df = pd.DataFrame(np.column_stack([y_train, x_train_flat]))
    test_df = pd.DataFrame(np.column_stack([y_test, x_test_flat]))
    train_df.to_csv(train_out, index=False, header=False)
    test_df.to_csv(test_out, index=False, header=False)
    
    if show_n > 0:
        n = int(show_n)
        cols = min(8, n)
        rows = int(np.ceil(n / cols))
        plt.figure(figsize=(2 * cols, 2 * rows))
        for i in range(n):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(x_train_norm[i], cmap="gray")
            plt.title(f"{y_train[i]}")
            plt.axis("off")
        plt.suptitle(f"First {n} normalized digits")
        plt.tight_layout()
        plt.show()


def binarize_mnist_to_csv(threshold=128, train_out="mnist_train_binarized.csv", test_out="mnist_test_binarized.csv", show_n=0):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    x_train_bin = (x_train_flat >= threshold).astype(np.uint8)
    x_test_bin = (x_test_flat >= threshold).astype(np.uint8)
    
    train_df = pd.DataFrame(np.column_stack([y_train, x_train_bin]))
    test_df = pd.DataFrame(np.column_stack([y_test, x_test_bin]))
    train_df.to_csv(train_out, index=False, header=False)
    test_df.to_csv(test_out, index=False, header=False)
    
    if show_n > 0:
        n = int(show_n)
        imgs = x_train_bin[:n].reshape(-1, 28, 28)
        labels = y_train[:n]
        cols = min(8, n)
        rows = int(np.ceil(n / cols))
        plt.figure(figsize=(2 * cols, 2 * rows))
        for i in range(len(imgs)):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
            plt.title(f"{labels[i]}")
            plt.axis("off")
        plt.suptitle(f"Binarized (thr={threshold}) - showing {n}")
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Task 1: Preprocessing MNIST to CSV")
    parser.add_argument("mode", choices=["normalize", "binarize"], help="Preprocessing mode")
    parser.add_argument("--train_out", default=None)
    parser.add_argument("--test_out", default=None)
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--show_n", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "normalize":
        normalize_mnist_to_csv(
            train_out=args.train_out or "mnist_train_normalized.csv",
            test_out=args.test_out or "mnist_test_normalized.csv",
            show_n=args.show_n
        )
    else:
        binarize_mnist_to_csv(
            threshold=args.threshold,
            train_out=args.train_out or "mnist_train_binarized.csv",
            test_out=args.test_out or "mnist_test_binarized.csv",
            show_n=args.show_n
        )


if __name__ == "__main__":
    main()