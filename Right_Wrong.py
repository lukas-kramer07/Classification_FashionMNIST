import os
from pathlib import Path
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the trained model
model_name = "Fashion_modelV4"
normalized = True
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
    if normalized:
        # Normalize the image
        img = img / 255.0
    # Reshape the image to (28, 28, 1)
    return img

def predict_image(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)
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




# Copy the respective image into a new folder Right or Wrong depending on the prediction. These folders are created for the respective model used for the prediction
def copy_image(image_path, model_name, is_correct):
    # Get the image file name and extension
    image_filename = os.path.basename(image_path)
    
    # Create the new folder path based on the model name and correctness
    new_folder = Path('Predictions_Right_Wrong', model_name, 'Right' if is_correct else 'Wrong')
    
    # Create the new folder if it doesn't exist
    new_folder.mkdir(parents=True, exist_ok=True)
    
    # Construct the new image path
    new_image_path = new_folder / image_filename
    
    # Copy the image to the new path
    shutil.copy(image_path, new_image_path)
    
    print(f"Image copied to: {new_image_path}")

# Computes the accuracy depending on the number of wrong and right predictions
def compute_accuracy(right_count, wrong_count):
    total_count = right_count + wrong_count
    return 0.0 if total_count == 0 else (right_count / total_count) * 100



#############
# Main function: Predicts every image form the image folder. The user then has to input whether the prediciton is right or wrong, after which the image is copied into a respective folder "Right" or "Wrong" for the model. 
# The accuracy of the model is determined using the number of wrong and right pictures
#############
def main():

    right_count = 0
    wrong_count = 0

    # Iterate over each file in the folder
    for filename in os.listdir(folder):
        image_path = os.path.join(folder, filename)
        
        #print(f"Image paths: {image_paths};")  //prints all the image paths
        print(f"Choosen image: {image_path}\n")

        #predict the label
        prediction = predict_image(image_path)
        print(f"\nPrediciton: {prediction}")


        #show preprocessed and original image
        #preprocessed image
        plt.figure(figsize=(9,9))
        plt.suptitle(model_name)
        plt.subplot(1,2,1)
        plt.imshow(preprocess_image(image_path), cmap=plt.cm.binary) 
        plt.title("Preprocessed image")
        plt.xlabel(f"Prediction: {prediction}")
        #original image
        plt.subplot(1,2,2)
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB) #opencv reads and displays an image as BGR format instead of RGB color format. Whereas matplotlib uses RGB color format to display image --> Converts from BGR to RGB
        plt.imshow(original_img)
        plt.title("Original image")
        plt.show()


        # user input whether the prediction is right or wrong
        user_input = input("Enter 'right' or 'wrong': ")

        if os.path.isfile(image_path):
            if user_input.lower() == "right" or user_input == "1":
                copy_image(image_path, model_name, True)
                right_count += 1
            elif user_input.lower() == "wrong" or user_input == "0":
                copy_image(image_path, model_name, False)
                wrong_count += 1
            else:
                print("Invalid input. Please enter 'right' or 'wrong'.")

    # Compute accuracy
    accuracy = compute_accuracy(right_count, wrong_count)
    print(f"Accuracy: {accuracy}%")



if __name__ == "__main__":
        main()