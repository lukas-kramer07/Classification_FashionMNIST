import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Load the Fashion MNIST dataset
(train_data, train_labels) , (test_data, test_labels) = fashion_mnist.load_data()

# Normalize the data
train_data_norm = train_data / 255.0
test_data_norm = test_data / 255.0

def add_noise(image):
    noise = np.random.normal(loc=0, scale=0.02, size=image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

# Create an instance of the ImageDataGenerator class for data augmentation
data_augmenter = ImageDataGenerator(
    rotation_range=10,  # rotate the image up to 10 degrees
    width_shift_range=0.05,  
    height_shift_range=0.05,  
    horizontal_flip=True,  # flip the image horizontally
    vertical_flip=False,  # do not flip the image vertically
    zoom_range=0.05,  # zoom in/out up to 5%
    fill_mode='nearest',  # fill gaps in the image with the nearest pixel
    preprocessing_function=add_noise  # Add the add_noise function as the preprocessing function
)

# Fit the data augmenter on the training data
data_augmenter.fit(tf.expand_dims(train_data_norm, 3))

# Display some of the original images
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(train_data_norm[i], cmap='binary')
    ax.set_title(f'Label: {train_labels[i]}')
plt.suptitle('Original Images')
plt.show()

# Display some of the augmented images
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    augmented_image, _ = data_augmenter.flow(
        tf.expand_dims(tf.expand_dims(train_data_norm[i], 0),-1),
        train_labels[i].reshape(1,)
    ).next()
    ax.imshow(augmented_image.squeeze(), cmap='binary')
    ax.set_title(f'Label: {train_labels[i]}')
plt.suptitle('Augmented Images')
plt.show()