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

### Fashion Model

This repository contains different versions of a fashion classification model implemented using TensorFlow. Each version has a different architecture and training configuration. The models are trained on the Fashion MNIST dataset, and their performance can be evaluated by running the provided scripts.

## Scripts

### 1. train.py

This script is used to train the fashion classification model on the Fashion MNIST dataset. It loads the training and validation data, constructs the model architecture, and trains the model using the specified configuration. After training, the script saves the trained model weights to the specified output directory.

#### Usage:

```python
!python train.py --data_dir <data_directory> --output_dir <output_directory> --version <model_version>
```

#### Arguments:
- `--data_dir`: The directory containing the Fashion MNIST dataset. The dataset should be stored in the format expected by the script.
- `--output_dir`: The directory where the trained model weights will be saved.
- `--version`: The version of the model architecture to use for training. Available versions: `v1`, `v2`, `v3`.

### 2. evaluate.py

This script is used to evaluate the performance of the trained fashion classification model on the test set of the Fashion MNIST dataset. It loads the test data and the trained model weights, constructs the model architecture, and calculates the accuracy of the model on the test data.

#### Usage:

```python
!python evaluate.py --data_dir <data_directory> --version <model_version>
```

#### Arguments:
- `--data_dir`: The directory containing the Fashion MNIST dataset. The dataset should be stored in the format expected by the script.
- `--version`: The version of the model architecture to use for evaluation. Available versions: `v1`, `v2`, `v3`.

### 3. predict.py

This script is used to make predictions using the trained fashion classification model on user-provided images. It loads the trained model weights, constructs the model architecture, and makes predictions on the provided images. The predicted class labels and probabilities are displayed.

#### Usage:

```python
!python predict.py --model <model_version> --image_file <image_file>
```

#### Arguments:
- `--model`: The version of the model architecture to use for prediction. Available versions: `v1`, `v2`, `v3`.
- `--image_file`: The file path to the input image for prediction.

### 4. visualize.py

This script is used to visualize the predictions made by the trained fashion classification model on user-provided images. It loads the trained model weights, constructs the model architecture, makes predictions on the provided images, and displays the images along with their predicted class labels and probabilities.

#### Usage:

```python
!python visualize.py --model <model_version> --image_file <image_file>
```

#### Arguments:
- `--model`: The version of the model architecture to use for visualization. Available versions: `v1`, `v2`, `v3`.
- `--image_file`: The file path to the input image for visualization.

---
## Conclusion

In this evaluation section, we have explored different aspects of the fashion classification models implemented in this repository. We started by training the models using the Fashion MNIST dataset and evaluated their performance on the test set. We then demonstrated how to use the trained models to make predictions on user-provided images and visualize the results. Through this evaluation, we have seen the effectiveness of the models in accurately classifying fashion items. The evaluation scripts provided in this repository enable easy training, evaluation, prediction, and visualization of the models. jects.

