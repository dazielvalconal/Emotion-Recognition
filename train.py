import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import os

# Đường dẫn tới thư mục train và test
train_dir = "train" # Thay đổi đường dẫn tới thư mục train
test_dir = "test"   # Thay đổi đường dẫn tới thư mục test


IMG_SIZE = (48, 48)

# Batch size
BATCH_SIZE = 32

# Tạo tập dữ liệu từ thư mục sử dụng tf.data API
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",  # Sử dụng ảnh grayscale
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=123,
)

# Lưu trữ class_names
class_names = train_dataset.class_names

# Tạo validation dataset
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",  
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False,
    seed=123,
)

# Áp dụng tăng cường dữ liệu và tối ưu hóa pipeline
AUTOTUNE = tf.data.AUTOTUNE

data_augmentation = Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
    ]
)


def preprocess(image, label):
    image = data_augmentation(image)
    image = image / 255.0  
    return image, label


train_dataset = train_dataset.map(preprocess, num_parallel_calls=AUTOTUNE)
validation_dataset = validation_dataset.map(
    lambda x, y: (x / 255.0, y), num_parallel_calls=AUTOTUNE
)

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)



def compute_class_weights_from_directory(directory, class_names):
    class_counts = {}
    total_samples = 0
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            num_files = len(os.listdir(class_dir))
            class_counts[i] = num_files
            total_samples += num_files
    class_weights = {}
    num_classes = len(class_names)
    for class_index, count in class_counts.items():
        class_weight = total_samples / (num_classes * count)
        class_weights[class_index] = class_weight
    return class_weights



class_weights_dict = compute_class_weights_from_directory(train_dir, class_names)


model = Sequential(
    [
        tf.keras.layers.InputLayer(
            input_shape=IMG_SIZE + (1,)
        ), 
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(len(class_names), activation="softmax"),
    ]
)


model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
)

model_checkpoint = ModelCheckpoint(
    "best_emotion_model_gray.h5",
    monitor="val_accuracy",
    save_best_only=True,
)


history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
)


model.save("final_emotion_model_gray.h5")


plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.show()
