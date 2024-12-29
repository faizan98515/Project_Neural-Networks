import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('CIFAR-10 Dataset: ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Normalize pixel values from [0,255] to [0,1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

print('After reshpaing: ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(16, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dense(100, activation='softmax')
])

model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,  # 20% of training data for validation
    epochs=20,
    batch_size=32,
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

import numpy as np
X_sample = X_test[0:1]
y_sample_true = np.argmax(y_test[0])
y_pred = model.predict(X_sample)
y_pred_label = np.argmax(y_pred[0])
print(f"Predicted Label: {y_pred_label}, True Label: {y_sample_true}")