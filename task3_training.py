import argparse
import numpy as np
import pandas as pd
import joblib
from typing import Tuple


def _load_xy_from_csv(train_csv, test_csv):
    train_df = pd.read_csv(train_csv, header=None)
    test_df = pd.read_csv(test_csv, header=None)
    y_train = train_df.iloc[:, 0].values
    x_train = train_df.iloc[:, 1:].values
    y_test = test_df.iloc[:, 0].values
    x_test = test_df.iloc[:, 1:].values
    return x_train, y_train, x_test, y_test


def train_cnn(train_csv="mnist_train_normalized.csv", test_csv="mnist_test_normalized.csv", model_out="mnist_cnn_model.keras"):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.utils import to_categorical
    from sklearn.metrics import accuracy_score

    print("Training CNN (Convolutional Neural Network)...")
    
    x_train, y_train_raw, x_test, y_test_raw = _load_xy_from_csv(train_csv, test_csv)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    y_train = to_categorical(y_train_raw, 10)
    y_test = to_categorical(y_test_raw, 10)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Model Architecture:")
    print(f"   - Input: 28x28x1 (grayscale images)")
    print(f"   - Conv2D layers: 32, 64 filters")
    print(f"   - Dense layers: 128, 10 (output)")
    print(f"   - Total parameters: {model.count_params():,}")
    
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
        ModelCheckpoint(model_out, save_best_only=True, monitor='val_accuracy'),
    ]
    
    print("Starting CNN Training...")
    print("   - Epochs: 15")
    print("   - Batch size: 128")
    print("   - Early stopping: patience=5")
    
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15, batch_size=128, callbacks=callbacks, verbose=1)
    
    print("Evaluating CNN...")
    preds = model.predict(x_test)
    test_acc = accuracy_score(y_test_raw, np.argmax(preds, axis=1))     # Get accuracy
    confidence = np.max(preds, axis=1)                                  # Get max probability for each sample
    avg_confidence = np.mean(confidence)                                # Get average confidence
    model.save(model_out)
    
    print("CNN Training Complete!")
    print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Average Confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
    print(f"   Saved to: {model_out}\n")


def train_rf(train_csv="mnist_train_normalized.csv", test_csv="mnist_test_normalized.csv", model_out="mnist_rf_model.joblib"):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    print("Training RF (Random Forest)...")
    
    x_train, y_train, x_test, y_test = _load_xy_from_csv(train_csv, test_csv)
    
    print("Model Configuration:")
    print("   - Trees: 100")
    print("   - Max depth: 20")
    print("   - Min samples split: 5")
    print("   - Min samples leaf: 2")
    print("   - Max features: sqrt")
    print("   - Parallel jobs: -1 (all cores)")
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    )
    
    print("Starting RF Training...")
    
    rf.fit(x_train, y_train)
    
    print("Evaluating RF...")
    preds = rf.predict(x_test)
    acc = accuracy_score(y_test, preds)     # Get accuracy
    proba = rf.predict_proba(x_test)        # Get probability for each sample
    confidence = np.max(proba, axis=1)      # Get max probability for each sample
    avg_confidence = np.mean(confidence)    # Get average confidence
    joblib.dump(rf, model_out)
    
    print("RF Training Complete!")
    print(f"   Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"   Average Confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
    print(f"   Saved to: {model_out}\n")


def train_svm(train_csv="mnist_train_normalized.csv", test_csv="mnist_test_normalized.csv", model_out="mnist_svm_model.joblib", scaler_out="mnist_svm_scaler.joblib"):
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    print("Training SVM (Support Vector Machine)...")
    
    x_train, y_train, x_test, y_test = _load_xy_from_csv(train_csv, test_csv)
    
    print("Model Configuration:")
    print("   - Kernel: RBF (Radial Basis Function)")
    print("   - C parameter: 1.0")
    print("   - Gamma: scale")
    print("   - Probability: True (for confidence scores)")
    print("   - Random state: 42")
    
    print("Preprocessing data...")
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    
    print("Starting SVM Training...")
    
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True, verbose=False)
    svm.fit(x_train_s, y_train)
    
    print("Evaluating SVM...")
    preds = svm.predict(x_test_s)
    acc = accuracy_score(y_test, preds)     # Get accuracy
    proba = svm.predict_proba(x_test_s)     # Get probability for each sample
    confidence = np.max(proba, axis=1)      # Get max probability for each sample
    avg_confidence = np.mean(confidence)    # Get average confidence
    joblib.dump(svm, model_out)
    joblib.dump(scaler, scaler_out)
    
    print("SVM Training Complete!")
    print(f"   Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"   Average Confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
    print(f"   Saved to: {model_out} and scaler to: {scaler_out}\n")


def main():
    parser = argparse.ArgumentParser(description="Train models on MNIST dataset")
    parser.add_argument("model", choices=["cnn", "rf", "svm", "all"])
    parser.add_argument("--train_csv", default="mnist_train_normalized.csv")
    parser.add_argument("--test_csv", default="mnist_test_normalized.csv")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    
    if args.model == "cnn":
        train_cnn(args.train_csv, args.test_csv, model_out=args.out or "mnist_cnn_model.keras")
    elif args.model == "rf":
        train_rf(args.train_csv, args.test_csv, model_out=args.out or "mnist_rf_model.joblib")
    elif args.model == "svm":
        train_svm(args.train_csv, args.test_csv, model_out=args.out or "mnist_svm_model.joblib")
    else:
        print("Training all models...")
        train_cnn(args.train_csv, args.test_csv, "mnist_cnn_model.keras")
        train_rf(args.train_csv, args.test_csv, "mnist_rf_model.joblib")
        train_svm(args.train_csv, args.test_csv, "mnist_svm_model.joblib")
        print("All models trained successfully!")


if __name__ == "__main__":
    main()