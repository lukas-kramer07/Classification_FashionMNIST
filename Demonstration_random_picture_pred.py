import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the trained model
model_name = "Fashion_modelV3"
model = tf.keras.models.load_model(model_name)

# Define the class names
class_names = ["T-shirt/top", "Trousers", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

# Get a list of image file paths in the folder
folder = "imageFolder"
image_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg") or f.endswith(".jpgeg")]

# Select a random image from the folder
random_image_path = random.choice(image_paths)


#function to crop and square an image to a specified size so that it doesn't become distorted
def crop_square(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h,w])

    # Centralize and crop
    crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]

    return cv2.resize(crop_img, (size, size), interpolation=interpolation)

# Function to preprocess the image
def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    # Resize the image to 28x28
    img = crop_square(img, 28)
    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    # Normalize the image
    img = img / 255.0
    # Reshape the image to (28, 28, 1)
    return img

def predict_image():
    # Preprocess the image
    img = preprocess_image(random_image_path)
    # Make a prediction
    y_prob = model.predict(img.reshape(1, 28,28)) 

    #print the probabilities
    print("Probabilities:\n")
    for i in range(len(class_names)):
        probability = float(tf.squeeze(y_prob)[i])*100 
        rounded_probability = np.round(probability, decimals=1)
        print(f"{class_names[i]}: {rounded_probability}%")

    y_pred = np.argmax(y_prob[0])
    # return the class name
    return class_names[y_pred]


def main():

    #print(f"Image paths: {image_paths};")  //prints all the image paths
    print(f"Choosen image: {random_image_path}\n")

    #predict the label
    prediction = predict_image()
    print(f"\nPrediciton: {prediction}")


    #show preprocessed and original image
    plt.figure(figsize=(9,9))
    plt.suptitle(model_name)
    plt.subplot(1,2,1)
    plt.imshow(preprocess_image(random_image_path), cmap=plt.cm.binary) 
    plt.title("Preprocessed image")
    plt.xlabel(f"Prediction: {prediction}")


    plt.subplot(1,2,2)
    original_img = cv2.imread(random_image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB) #opencv reads and displays an image as BGR format instead of RGB color format. Whereas matplotlib uses RGB color format to display image --> Converts from BGR to RGB
    plt.imshow(original_img)
    plt.title("Original image")
    plt.show()

if __name__ == "__main__":
    main()