# Fashion Model README

This repository contains several versions of a fashion classification model implemented using TensorFlow. Each version is represented by a separate script: V1.py, V2.py, V3.py, and V4.py.


# Models

## Dataset
The models are trained on the Fashion MNIST dataset, which consists of 60,000 grayscale images of 10 different clothing items. The dataset is already split into training and test sets.

## Setup

1. Create an image folder:
   - Create a folder named "images" in the same directory as the script files.
   - Download the Fashion MNIST dataset and extract the contents.
   - Move the extracted train-images-idx3-ubyte and t10k-images-idx3-ubyte files to the "images" folder.

2. Training the models:
   - Open Google Colab and create a new notebook.
   - Copy the code from the desired version (V1.py, V2.py, V3.py, or V4.py) into separate cells in the notebook.
   - Run each cell in order to train the model.

3. Saving the models:
   - After training, the models will be saved in the current directory with the names "Fashion_modelV1", "Fashion_modelV2", "Fashion_modelV3", and "Fashion_modelV4" for V1.py, V2.py, V3.py, and V4.py, respectively.

4. Displaying results:
   - Each script includes a function to create a confusion matrix and display the results.
   - After training, run the corresponding cell in the notebook to visualize the confusion matrix and plot the training history.

## Version Details

### V1.py
- Architecture: 
  - Input layer: Flatten (input_shape=(28, 28))
  - Hidden layers: Dense(100, activation="relu") -> Dense(100, activation="relu") -> Dense(50, activation="relu")
  - Output layer: Dense(10, activation="softmax")
- Training: 
  - Loss function: sparse_categorical_crossentropy
  - Optimizer: Adam
  - Metrics: accuracy

### V2.py
- Architecture:
  - Input layer: Flatten (input_shape=(28, 28))
  - Hidden layers: Dense(250, activation="relu") -> Dense(150, activation="relu") -> Dense(250, activation="relu") -> Dense(150, activation="relu") -> Dense(300, activation="relu")
  - Output layer: Dense(10, activation="softmax")
- Training:
  - Loss function: sparse_categorical_crossentropy
  - Optimizer: Adam
  - Metrics: accuracy

### V3.py
- Architecture:
  - Input layer: Flatten (input_shape=(28, 28))
  - Hidden layers: Dense(250, activation="relu") -> Dense(150, activation="relu") -> Dense(250, activation="relu") -> Dense(150, activation="relu") -> Dense(300, activation="relu")
  - Output layer: Dense(10, activation="softmax")
- Training:
  - Loss function: sparse_categorical_crossentropy
  - Optimizer: Adam
  - Metrics: accuracy
- Data Augmentation:
  - ImageDataGenerator used for augmentation:
    - Rotation range: 10 degrees
    - Width shift range: 0.05
    - Height shift range: 0.05
    - Horizontal flip: True
    - Vertical flip: False
    - Zoom range: 0.05
    - Fill mode: nearest
    - Preprocessing function: add_noise (adds random noise to images)

### V4.py
- Architecture:
  - Input layer: Flatten (input_shape=(28, 28))
- Hidden layers: Dense(256, activation="relu") -> Dense(128, activation="relu") -> Dense(64, activation="relu")
  - Output layer: Dense(10, activation="softmax")
- Training:
  - Loss function: sparse_categorical_crossentropy
  - Optimizer: Adam
  - Metrics: accuracy
- Regularization:
  - Dropout regularization applied to the hidden layers with a rate of 0.3

## Conclusion - Models
The Fashion Model repository provides different versions of fashion classification models implemented using TensorFlow. Each version has a different architecture and training configuration. By following the setup instructions and running the provided scripts, you can train the models on the Fashion MNIST dataset and evaluate their performance. Feel free to experiment with different versions and configurations to improve the model's accuracy and explore different architectural choices.


# Evaluation

## Fashion Model Evaluation Scripts

This section provides documentation for the evaluation scripts included in the Fashion Model repository. These scripts are designed to evaluate the performance of the trained fashion classification models, demonstrate different features, and make predictions on your own pictures. Below you will find details on each script and how to use them.

### Classification.py

This script is used to evaluate the performance of the trained fashion classification models on the Fashion MNIST test dataset. It loads the saved model weights and evaluates the accuracy of the model on the test data. The script provides a command-line interface to specify the model version to evaluate.

**Arguments:**
- `--version`: Specifies the model version to evaluate (e.g., "v1", "v2", etc.)
- `--image_folder_path`: Specifies the path to the folder containing the test images.

### Demonstration_DataAugmenter.py

This script demonstrates the data augmentation capabilities of the trained fashion classification models. It generates augmented images by applying various transformations to the original images from the Fashion MNIST dataset. The script saves the augmented images to a specified directory for visual inspection.


### Demonstration_random_picture_pred.py

This script allows you to make predictions on your own pictures using the trained fashion classification models. It takes an input image and applies pre-processing steps to make it compatible with the model's input requirements. The script then loads the saved model weights and predicts the class label for the input image.

**Arguments:**
- `--version`: Specifies the model version to evaluate (e.g., "v1", "v2", etc.)
- `--image_folder_path`: Specifies the path to the folder containing the input images for prediction.

### Right_Wrong.py

This script is used to evaluate the model's predictions on the test dataset and generate a report indicating the correctly and wrongly classified images. It loads the saved model weights, performs predictions on the test data, and generates an HTML report that displays the test images along with their predicted and true labels.

**Arguments:**
- `--version`: Specifies the model version to evaluate (e.g., "v1", "v2", etc.)
- `--image_folder_path`: Specifies the path to the folder containing the test images.

**Note:** Please make sure to modify the script arguments and usage as necessary depending on your specific environment, such as Google Colab.

---

Please refer to the repository's Readme file for detailed instructions on how to set up and use these scripts to evaluate the fashion classification models and explore their features.
## Conclusion

In this evaluation section, we have explored different aspects of the fashion classification models implemented in this repository. We started by training the models using the Fashion MNIST dataset and evaluated their performance on the test set. We then demonstrated how to use the trained models to make predictions on user-provided images and visualize the results. Through this evaluation, we have seen the effectiveness of the models in accurately classifying fashion items. The evaluation scripts provided in this repository enable easy training, evaluation, prediction, and visualization of the models. jects.

