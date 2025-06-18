import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# 1. Paths to data
image_dir = "data/homeobjects-3K/train/images"
label_dir = "data/homeobjects-3K/train/labels"

class_names = [
    "bed", "sofa", "chair", "table", "lamp", "tv", "laptop",
    "wardrobe", "window", "door", "potted plant", "photo frame"
]
NUM_CLASSES = len(class_names)

# 2. Read and crop training data
X = []
y = []

for img_name in sorted(os.listdir(image_dir))[:300]:  # Limit for speed
    img_path = os.path.join(image_dir, img_name)
    label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))

    if not os.path.exists(label_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w, _ = img.shape

    with open(label_path, "r") as f:
        for line in f:
            class_id, x_center, y_center, box_w, box_h = map(float, line.strip().split())
            class_id = int(class_id)

            # Convert YOLO box to pixel coordinates
            x1 = int((x_center - box_w / 2) * w)
            y1 = int((y_center - box_h / 2) * h)
            x2 = int((x_center + box_w / 2) * w)
            y2 = int((y_center + box_h / 2) * h)

            # Crop and resize to fixed shape
            cropped = img[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            resized = cv2.resize(cropped, (64, 64))
            X.append(resized)
            y.append(class_id)

# 3. Convert to NumPy arrays
X = np.array(X) / 255.0  # Normalize pixels to 0–1
y = tf.keras.utils.to_categorical(y, NUM_CLASSES)  # One-hot encode

# 4. Split into training and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Build a CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. Train the model
model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))
