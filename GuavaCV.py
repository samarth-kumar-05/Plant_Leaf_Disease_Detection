import cv2
from tensorflow import keras
import numpy as np
import keyboard
import os

import tensorflow as tf
from tensorflow.keras.preprocessing import image as img_preprocessing
import numpy as np

from rembg import remove 
from PIL import Image 

# Function to preprocess the input image
# def preprocess_image(image_array, target_size=(224, 224)):
#     img_array = cv2.resize(image_array, target_size)  # Resize image
#     img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
#     img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for model input
#     return img_array

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


# Function to print the class with the highest probability
def print_predictions(predictions, class_names):
    max_prob_index = np.argmax(predictions)
    max_prob_class = class_names[max_prob_index]
    max_prob_value = predictions[0][max_prob_index]
    print(f"Predicted class: {max_prob_class}, Probability: {max_prob_value}")

# Function to save the preprocessed image
def save_preprocessed_image(image_array, output_path):
    # Convert the image array to PIL Image
    image_pil = Image.fromarray(image_array.astype('uint8'), 'RGB')
    # Save the image
    image_pil.save(output_path)

# Load the saved model and define categories
loaded_model = keras.models.load_model('guavaEfficientNet.h5')
class_names = ['Guava_Disease','Guava_Healthy'] 

# Initialize camera
cap = cv2.VideoCapture(0)

# Create a folder to store the images if it doesn't exist
output_folder = 'captured_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

while True:
    # Display the camera feed
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Press s to capture photo', frame)

    # Check for key press to capture photo
    if keyboard.is_pressed('s'):
        # Capture photo
        ret, photo = cap.read()
        if ret:
            # Save the captured photo
            photo_path = os.path.join(output_folder, 'captured_photo.jpg')
            cv2.imwrite(photo_path, photo)
            
            # Remove background and predict leaf disease

            print(photo_path)
            preprocessed_img = preprocess_image(photo_path)
            # predicted_class = predict_image(loaded_model, preprocessed_img)
            # print(f"Predicted Class: {predicted_class}")
            # print_predictions(predicted_class, categories)
            predictions = predict_image(loaded_model, photo_path)
            print_predictions(predictions, class_names)

            # Save the preprocessed image
            preprocessed_photo_path = os.path.join(output_folder, 'preprocessed_photo.jpg')
            save_preprocessed_image(preprocessed_img.squeeze(), preprocessed_photo_path)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
