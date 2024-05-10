import serial
import time
import os

def select_plant():
    print("Select the plant for leaf disease detection:")
    print("1. Guava")
    print("2. Hibiscus")
    print("3. Cotton")
    choice = input("Enter your choice (1/2/3): ")
    return choice

def execute_selected_plant(choice):
    if choice == '1':
        os.system('python guava.py')
    elif choice == '2':
        os.system('python hibiscus.py')
    elif choice == '3':
        os.system('python cotton.py')
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    ser = serial.Serial('COM10', 9600)  # Adjust COM port accordingly
    time.sleep(2)  # Allow time for serial communication to establish
    
    plant_choice = select_plant()
    ser.write(plant_choice.encode())

    
    # Wait for 2 seconds to display the selected plant choice on the LCD
    time.sleep(2)  

    execute_selected_plant(plant_choice)
    
    # Assuming you have the prediction result stored in a variable named 'prediction_result'
    # Read the prediction result from the text file
    prediction_result = ""
    with open('prediction_result.txt', 'r') as file:
        prediction_result = file.read().strip()
      
    
    ser.write(prediction_result.encode())  # Send the prediction result to Arduino
    time.sleep(2)  # Wait for 2 seconds to display the prediction result on the LCD
