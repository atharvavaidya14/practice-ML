import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = (
    cifar10.load_data()
)  # tensorflow dataset object
train_images, test_images = train_images / 255.0, test_images / 255.0  # Normalizing

class_names = [
    "airplanes",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
inputs = np.concatenate((train_images, test_images), axis=0)
targets = np.concatenate((train_labels, test_labels), axis=0)

kfold = KFold(n_splits=5, random_state=7, shuffle=True)
acc_per_fold = []
loss_per_fold = []
fold = 1

for train_index, test_index in kfold.split(inputs, targets):
    # CNN Architecture
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    # Dense Layers
    model.add(layers.Flatten())  # flatten the result for dense layers
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(10))

    model.summary()
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    print(f"Training for fold {fold} ...")
    history = model.fit(
        train_images, train_labels, epochs=8, validation_data=(test_images, test_labels)
    )
    scores = model.evaluate(inputs[test_index], targets[test_index], verbose=0)
    print(scores)
    print(
        f"Score for fold {fold}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%"
    )
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold += 1

    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.5, 1])
    plt.legend(loc="lower right")

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print(test_acc)
