import argparse
import os
import glob
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from typing import List, Tuple

from task2_segmentation import segment_image, extract_digits


def _load_test_csv(test_csv):
    df = pd.read_csv(test_csv, header=None)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    return X, y


def eval_cnn(test_csv="mnist_test_normalized.csv", model_path="mnist_cnn_model.keras"):
    import tensorflow as tf
    
    X, y = _load_test_csv(test_csv)
    X_img = X.reshape(X.shape[0], 28, 28, 1)
    model = tf.keras.models.load_model(model_path)
    probs = model.predict(X_img, verbose=0)
    pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y, pred)
    # Calculate confidence
    confidence = np.max(probs, axis=1)
    avg_confidence = np.mean(confidence)
    print(f"CNN Average Confidence: {avg_confidence:.4f}")
    _show_confusion(y, pred, title=f"CNN ({model_path})")
    return acc


def eval_rf(test_csv="mnist_test_normalized.csv", model_path="mnist_rf_model.joblib"):
    X, y = _load_test_csv(test_csv)
    model = joblib.load(model_path)
    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    # Calculate confidence
    proba = model.predict_proba(X)
    confidence = np.max(proba, axis=1)
    avg_confidence = np.mean(confidence)
    print(f"RF Average Confidence: {avg_confidence:.4f}")
    _show_confusion(y, pred, title=f"RF ({model_path})")
    return acc


def eval_svm(test_csv="mnist_test_normalized.csv", model_path="mnist_svm_model.joblib", scaler_path="mnist_svm_scaler.joblib"):
    X, y = _load_test_csv(test_csv)
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    Xs = scaler.transform(X)
    pred = model.predict(Xs)
    acc = accuracy_score(y, pred)
    # Calculate confidence using predict_proba (requires probability=True in training)
    proba = model.predict_proba(Xs)
    confidence = np.max(proba, axis=1)
    avg_confidence = np.mean(confidence)
    print(f"SVM Average Confidence: {avg_confidence:.4f}")
    _show_confusion(y, pred, title=f"SVM ({model_path})")
    return acc


def _show_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    print(title)
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {title}")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def _eval_folder_common(folder: str, predictor, method: str = "auto", one: bool = True) -> float:
    paths = sorted(glob.glob(os.path.join(folder, "*.png")))
    y_true: List[int] = []
    y_pred: List[int] = []
    for path in paths:
        base = os.path.basename(path)
        try:
            true_label = int(base.split('_')[1])
        except Exception:
            continue
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        mask = segment_image(gray, method=method)
        chips, _ = extract_digits(mask, gray, one_digit_only=one)
        if not chips:
            continue
        pred = predictor(chips[0])
        y_true.append(true_label)
        y_pred.append(pred)
    if not y_true:
        print("No evaluable images in folder")
        return 0.0
    _show_confusion(np.array(y_true), np.array(y_pred), title=f"Folder eval ({method}, one={one})")
    return accuracy_score(y_true, y_pred)


def eval_folder_cnn(folder: str, model_path: str = "mnist_cnn_model.keras", method: str = "auto", one: bool = True) -> float:
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    def _pred(chip28: np.ndarray) -> int:
        probs = model.predict(chip28.reshape(1,28,28,1), verbose=0)
        return int(np.argmax(probs[0]))
    return _eval_folder_common(folder, _pred, method=method, one=one)


def eval_folder_rf(folder: str, model_path: str = "mnist_rf_model.joblib", method: str = "auto", one: bool = True) -> float:
    model = joblib.load(model_path)
    def _pred(chip28: np.ndarray) -> int:
        return int(model.predict(chip28.reshape(1,-1))[0])
    return _eval_folder_common(folder, _pred, method=method, one=one)


def eval_folder_svm(folder: str, model_path: str = "mnist_svm_model.joblib", scaler_path: str = "mnist_svm_scaler.joblib", method: str = "auto", one: bool = True) -> float:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    def _pred(chip28: np.ndarray) -> int:
        vec = chip28.reshape(1,-1)
        vecs = scaler.transform(vec)
        return int(model.predict(vecs)[0])
    return _eval_folder_common(folder, _pred, method=method, one=one)


def main():
    parser = argparse.ArgumentParser(description="Task 3: Evaluate trained models on test CSV")
    parser.add_argument("model", choices=["cnn", "rf", "svm", "all"])
    parser.add_argument("--test_csv", default="mnist_test_normalized.csv")
    parser.add_argument("--cnn_path", default="mnist_cnn_model.keras")
    parser.add_argument("--rf_path", default="mnist_rf_model.joblib")
    parser.add_argument("--svm_path", default="mnist_svm_model.joblib")
    parser.add_argument("--svm_scaler", default="mnist_svm_scaler.joblib")
    args = parser.parse_args()

    if args.model == "cnn":
        acc = eval_cnn(args.test_csv, args.cnn_path)
        print(f"CNN accuracy: {acc:.4f}")
    elif args.model == "rf":
        acc = eval_rf(args.test_csv, args.rf_path)
        print(f"RF accuracy: {acc:.4f}")
    elif args.model == "svm":
        acc = eval_svm(args.test_csv, args.svm_path, args.svm_scaler)
        print(f"SVM accuracy: {acc:.4f}")
    else:
        results = {
            "cnn": eval_cnn(args.test_csv, args.cnn_path),
            "rf": eval_rf(args.test_csv, args.rf_path),
            "svm": eval_svm(args.test_csv, args.svm_path, args.svm_scaler),
        }
        print("\nSummary (accuracy):")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()