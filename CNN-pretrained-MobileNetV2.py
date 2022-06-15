import keras.layers
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import tensorflow as tf
import tensorflow_datasets as ds

(og_train, og_validation, og_test), metadata = ds.load('cats_vs_dogs',
                                                       split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
                                                       with_info=True, as_supervised=True)

label_name = metadata.features['label'].int2str  # function object to get labels
print(type(label_name))
# for img, label in og_train.take(2):
#     plt.figure()
#     plt.imshow(img)
#     plt.title(label_name(label))
#     plt.show()
#     print("Old shape is ", img.shape)

size = 160


def format_img(image, label):
    image = tf.cast(image, tf.float32)  # Casts all values in the image to float32. they can be integers
    image = (image / 127.5) - 1  # 255/2
    image = tf.image.resize(image, (size, size))
    return image, label


train = og_train.map(format_img)  # map applies the function in the () to every example in og_train
validation = og_validation.map(format_img)
test = og_test.map(format_img)

# for img, label in train.take(2):
#     plt.figure()
#     plt.imshow(img, cmap='Spectral')
#     plt.title(label_name(label))
#     plt.show()
#     print("new shape is", img.shape)

# Shuffle and Batch the images
batch_size = 32
shuffle_buffer_size = 1000

train_batches = train.shuffle(shuffle_buffer_size).batch(batch_size)
validation_batches = validation.batch(batch_size)
test_batches = test.batch(batch_size)
# Using a pre-trained CNN- MobileNet V2

model = tf.keras.applications.MobileNetV2(input_shape=(size, size, 3), include_top=False, weights='imagenet')

# include_top specifies whether to include the classifier of this CNN. we want to retrain this CNN only on cats snd dogs
# So just two classes
# weights are the learned weights from imagenet (i guess)
model.summary()

for image, _ in train_batches.take(1):
    pass
feature_batch = model(image)
print(feature_batch.shape)

# Freezing the base model. We don't want to change the pre-learned weights
model.trainable = False

# Adding my classifier
global_average = tf.keras.layers.GlobalAveragePooling2D()
# This will (instead of flattening) take the average of the entire 5x5 area of each 2D feature map and return a 1280
# element vector per filter

classifier = keras.layers.Dense(1)
final_model = tf.keras.Sequential([model, global_average, classifier])
model.summary()

# Training the model
alpha = 0.0001  # learning rate
final_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=alpha),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
# Training the model on the cats vs dogs images
history = final_model.fit(train_batches, epochs=3, validation_data=validation_batches)
accuracy = history.history['accuracy']
print(accuracy)

final_model.save("cats_vs_dogs.h5")  # keras specific extension

new_model = tf.keras.models.load_model('cats_vs_dogs.h5')
