import os
from pathlib import Path
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the trained model
model_name = "Fashion_modelV3"
model = tf.keras.models.load_model(model_name)

# Define the class names
class_names = ["T-shirt/top", "Trousers", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

# Define the image folder
folder = "imageFolder"

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

def predict_image(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)
    # Make a prediction
    y_prob = model.predict(img.reshape(1, 28,28))                                                                                                                                                                                                                           # type: ignore

    #print the probabilities
    print("Probabilities:\n")
    for i in range(len(class_names)):
        probability = float(tf.squeeze(y_prob)[i])*100                                                                                                                                                                                                                      # type: ignore
        rounded_probability = np.round(probability, decimals=1)
        print(f"{class_names[i]}: {rounded_probability}%")

    y_pred = np.argmax(y_prob[0])
    # return the class name
    return class_names[y_pred]




# Copy the respective image into a new folder Right or Wrong depending on the prediction. These folders are created for the respective model used for the prediction
def copy_image(image_path, model_name, label):
    # Get the image file name and extension
    image_filename = os.path.basename(image_path)

    label = label.replace("/","_") #"T-shirt/Top" can't be a folder name
    
    # Create the new folder path based on the model name and correctness
    new_folder = Path('Predictions_allClasses', model_name, label)
    
    # Create the new folder if it doesn't exist
    new_folder.mkdir(parents=True, exist_ok=True)
    
    # Construct the new image path
    new_image_path = new_folder / image_filename
    
    # Copy the image to the new path
    shutil.copy(image_path, new_image_path)
    
    print(f"Image copied to: {new_image_path}")


#############
# Main function: Predicts every image form the image folder and creates a copy in a respective folder
#############
def main():

    # Iterate over each file in the folder
    for filename in os.listdir(folder):
        image_path = os.path.join(folder, filename)
        print(f"Choosen image: {image_path}\n")

        #predict the label
        prediction = predict_image(image_path)
        print(f"\nPrediciton: {prediction}")

        if os.path.isfile(image_path):
            copy_image(image_path, model_name, prediction)



if __name__ == "__main__":
    main()