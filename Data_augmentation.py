import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets
import matplotlib.pyplot as plt

data = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
test = train_images[np.random.randint(0, len(train_images))]
img = image.img_to_array(
    test
)  # Converting the dataset object to a numpy array. 32, 32, 3
print(np.shape(img))
img = img.reshape((1,) + img.shape)  # adding a dimension of one. 1, 32, 32, 3
print(np.shape(img))

i = 0
for batch in data.flow(img, save_prefix="test", save_format="jpeg"):
    plt.figure(i)
    plot = plt.imshow(image.img_to_array(batch[0]))
    i += 1
    if i > 4:
        break

plt.show()
