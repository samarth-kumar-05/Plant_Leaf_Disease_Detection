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
    plant_choice = select_plant()
    execute_selected_plant(plant_choice)
