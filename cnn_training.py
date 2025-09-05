import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# --- Config ---
TRAIN_CSV = 'mnist_train_normalized.csv'
TEST_CSV = 'mnist_test_normalized.csv'
MODEL_OUT = 'mnist_cnn_model.keras'

# --- 1. Load and Prepare Data ---
print("--- Loading and Preparing Data ---")
try:
    train_df = pd.read_csv(TRAIN_CSV, header=None)
    test_df = pd.read_csv(TEST_CSV, header=None)
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure the CSV files are in the same directory.")
    exit()

# Separate labels (first column) from pixels (the rest)
y_train_raw = train_df.iloc[:, 0].values
x_train_flat = train_df.iloc[:, 1:].values

y_test_raw = test_df.iloc[:, 0].values
x_test_flat = test_df.iloc[:, 1:].values

# Reshape the flat pixel data (784 columns) into 28x28x1 images for the CNN
x_train = x_train_flat.reshape(x_train_flat.shape[0], 28, 28, 1)
x_test = x_test_flat.reshape(x_test_flat.shape[0], 28, 28, 1)

# One-hot encode the labels (e.g., 5 -> [0,0,0,0,0,1,0,0,0,0])
y_train = to_categorical(y_train_raw, 10)
y_test = to_categorical(y_test_raw, 10)

print(f"Data Loaded. Training images shape: {x_train.shape}")
print(f"Data Loaded. Test images shape: {x_test.shape}")


# --- 2. Build the CNN Model ---
print("\n--- Building CNN Model ---")
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# --- 3. Train the Model ---
print("\n--- Training Model ---")
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
    ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor='val_accuracy')
]

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=15,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

model.save(MODEL_OUT)
print(f"\n✅ Model saved to {MODEL_OUT}")


# --- 4. Visualize Training History ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot training & validation accuracy values
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('Model Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend(['Train', 'Test'], loc='upper left')
ax1.grid(True)

# Plot training & validation loss values
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model Loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['Train', 'Test'], loc='upper left')
ax2.grid(True)

plt.tight_layout()
plt.show()


# --- 5. Visualize Model Predictions ---
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(10, 8))
for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    
    title_color = 'g' if predicted_labels[i] == y_test_raw[i] else 'r'
    plt.title(f"Pred: {predicted_labels[i]}\nTrue: {y_test_raw[i]}", color=title_color)
    plt.axis('off')

plt.tight_layout()
plt.show()
