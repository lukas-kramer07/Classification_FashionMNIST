import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Create a confusion matrix
# Note: Adapted from scikit-learn's plot_confusion_matrix()

import itertools
from sklearn.metrics import confusion_matrix

class_names = ["T-shirt/top","Trousers", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10,10,), text_size=15):
  #Create the confusion matrix
  cm = confusion_matrix(y_true, tf.round(y_preds))
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
  n_classes= cm.shape[0]

  # Let's prettify it
  fig, ax = plt.subplots(figsize=figsize)
  #Create a matrix plot
  cax = ax.matshow(cm, cmap=plt.cm.Purples)
  fig.colorbar(cax)

#set labels to be classes
  labels = classes if classes else np.arange(cm.shape[0])
  # Label the axes
  ax.set(title="Confusion matrix - Fashion_modelV3",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels)

  # Set x-labels to the bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  #Adjust label size
  ax.xaxis.label.set_size(text_size)
  ax.yaxis.label.set_size(text_size)
  ax.title.set_size(text_size)

  # Set coluor threshhold
  threshold = (cm.max() + cm.min())/2.

  #Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]} ({cm_norm[i,j]*100:.1f}%)",
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else "black",
            size=text_size)







# The data has already been sorted into training and test sets 
(train_data, train_labels) , (test_data, test_labels) = fashion_mnist.load_data()
# We can normalize our training data by deviding by the max (255)
train_data_norm = train_data/255
test_data_norm = test_data/255
tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(250, activation="relu"),
    tf.keras.layers.Dense(150, activation="relu"),
    tf.keras.layers.Dense(250, activation="relu"),
    tf.keras.layers.Dense(150, activation="relu"),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
                 optimizer=tf.keras.optimizers.Adam(),
                 metrics=["accuracy"])

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

# Train the model on the augmented data
augmented_history = model.fit(
    data_augmenter.flow(tf.expand_dims(train_data_norm, -1), train_labels),
    epochs=50,
    validation_data=(tf.expand_dims(test_data_norm, -1), test_labels),
    )


model.save("Fashion_modelV3")

y_probs = model.predict(test_data_norm)
y_preds = tf.argmax(y_probs, axis=1)


make_confusion_matrix(y_true=test_labels,
                      y_pred=y_preds,
                      classes=class_names,
                      figsize=(13,13),
                      text_size=8)
pd.DataFrame(augmented_history.history).plot(title="Fashion_modelV3")
plt.show()