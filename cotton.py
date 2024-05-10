TF_ENABLE_ONEDNN_OPTS=0
import cv2
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import keyboard
import os
import uuid


import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as img_preprocessing
import numpy as np

from rembg import remove 
from PIL import Image 

# Function to preprocess the input image
def preprocess_image(image_path, target_size=(224, 224)):
    img = img_preprocessing.load_img(image_path, target_size=target_size)
    img_array = img_preprocessing.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict_image(model, image_path, target_size=(224, 224)):
    img_array = preprocess_image(image_path, target_size=target_size)
    # Preprocess input for the model
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    # Get predictions
    predictions = model.predict(img_array)
    return predictions

# Function to print output predictions
def print_predictions(predictions, class_names):
    for i, prob in enumerate(predictions[0]):
        print(f"{class_names[i]}: {prob}")
# Function to print the class with the highest probability

def print_predictions(predictions, class_names):
    max_prob_index = np.argmax(predictions)
    max_prob_class = class_names[max_prob_index]
    max_prob_value = predictions[0][max_prob_index]
    print(f"Predicted class: {max_prob_class}, Probability: {max_prob_value}")

def save_preprocessed_image(image_array, output_folder):
    # Generate a unique file name for the image
    image_filename = str(uuid.uuid4()) + '.png'
    output_path = os.path.join(output_folder, image_filename)
    # Convert the image array to PIL Image
    image_pil = Image.fromarray(image_array.astype('uint8'), 'RGB')
    # Save the image
    image_pil.save(output_path)
    return output_path  # Return the path of the saved image

# Function to store the prediction result in a text file
def store_prediction_result(prediction_result, filename='prediction_result.txt'):
    with open(filename, 'w') as file:
        file.write(prediction_result)

# Load the saved model and define categories
loaded_model = keras.models.load_model('cottonEfficientNet.h5')
categories = ['Cotton_Disease','Cotton_Healthy'] 

# Initialize camera
cap = cv2.VideoCapture(0)

# Create a folder to store the images if it doesn't exist
output_folder = 'captured_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Set up matplotlib figure and axes
plt.ion()
fig, ax = plt.subplots()

while True:
    # Display the camera feed
    ret, frame = cap.read()
    if ret:
        ax.clear()
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title("Press 's' to capture photo")
        ax.axis('off')
        plt.pause(0.01)  # Pause to update the plot

    # Check for key press to capture photo
    if keyboard.is_pressed('w'):
        # Capture photo
        ret, photo = cap.read()
        if ret:
            # Save the captured photo
            # photo_path = os.path.join(output_folder, 'captured_photo.png')
            # cv2.imwrite(photo_path, photo)

            # Remove background and predict leaf disease
            photo_path = save_preprocessed_image(photo, output_folder)
            predicted_class = predict_image(loaded_model, photo_path)
            
            # Remove background and predict leaf disease
            # photo_path = './captured_images/Figure_1.png'
            # predicted_class = predict_image(loaded_model, photo_path)

            max_prob_index = np.argmax(predicted_class)
            max_prob_class = categories[max_prob_index]
            max_prob_value = predicted_class[0][max_prob_index]
            print(f"Predicted class: {max_prob_class}, Probability: {max_prob_value}")
            print(f"Predicted Class: {predicted_class}")
            
            # # Save the preprocessed image
            # preprocessed_img = preprocess_image(photo_path)
            # preprocessed_photo_path = os.path.join(output_folder, 'preprocessed_photo.png')
            # save_preprocessed_image(preprocessed_img.squeeze(), preprocessed_photo_path)
            
            # print(f"Predicted Class: {predicted_class}")

    # Break the loop when 'q' is pressed
    if keyboard.is_pressed('q'):
        break

# Release the camera
cap.release()
plt.ioff()
plt.show()

            # Remove background and predict leaf disease
photo_path = './captured_images/Figure_1.png'
predicted_class = predict_image(loaded_model, photo_path)

max_prob_index = np.argmax(predicted_class)
max_prob_class = categories[max_prob_index]
max_prob_value = predicted_class[0][max_prob_index]
print(f"Predicted class: {max_prob_class}, Probability: {max_prob_value}")
print(f"Predicted Class: {predicted_class}")

store_prediction_result(max_prob_class)