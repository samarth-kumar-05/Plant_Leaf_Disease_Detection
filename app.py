import cv2
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import keyboard
import os

from rembg import remove 
from PIL import Image 

def preprocess_image(img, target_size=(224, 224)):
    # Resize image to target size
    img = cv2.resize(img, target_size)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply background subtraction (example using MOG2)
    fgmask = cv2.createBackgroundSubtractorMOG2().apply(gray)
    
    # Threshold mask to get binary image
    _, binary_mask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
    
    # Invert mask
    binary_mask = cv2.bitwise_not(binary_mask)
    
    # Apply mask to original image
    img = cv2.bitwise_and(img, img, mask=binary_mask)
    
    # Normalize pixel values to [0, 1]
    img = img / 255.0

    img = remove(img) 
    
    return img

def predict_leaf_disease(model, img, categories):
    img = preprocess_image(img)
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)
    predicted_class = categories[predicted_class_index]
    return predicted_class

# Load the saved model and define categories
loaded_model = keras.models.load_model('leaf-cnn.h5')
categories = ['Guava_Healthy','Giava_Disease', 'Cotton_Disease', 'Cotton_Healthy','Hibiscus_Healthy', 'Hibiscus_diseased']

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
    if keyboard.is_pressed('s'):
        # Capture photo
        ret, photo = cap.read()
        if ret:
            # Save the captured photo
            photo_path = os.path.join(output_folder, 'captured_photo.jpg')
            cv2.imwrite(photo_path, photo)
            
            # Remove background and predict leaf disease
            predicted_class = predict_leaf_disease(loaded_model, photo, categories)
            
            # Save the preprocessed image
            preprocessed_img = preprocess_image(photo)
            preprocessed_photo_path = os.path.join(output_folder, 'preprocessed_photo.jpg')
            cv2.imwrite(preprocessed_photo_path, preprocessed_img*255)
            
            print(f"Predicted Class: {predicted_class}")

    # Break the loop when 'q' is pressed
    if keyboard.is_pressed('q'):
        break

# Release the camera
cap.release()
plt.ioff()
plt.show()
