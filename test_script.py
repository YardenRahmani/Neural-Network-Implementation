import subprocess
import os
import random

LEARNING_RATE = 0.001
HIDDEN_LAYER_SIZE = [4,8,4]
SAMPLES = 1000
EPOCHES = 10000
consts = [str(LEARNING_RATE), str(HIDDEN_LAYER_SIZE), str(EPOCHES)]

def generate_data(testNum):
    fileName = "data" + str(testNum)
    with open(fileName, "w") as dataFile:
        if testNum == 1:
            inputSize = 1
        elif testNum == 2:
            inputSize = 2
        outputSize = 1
        for _ in range(SAMPLES):
            x1 = 5*random.random()
            if testNum == 1:
                dataFile.write(f"{x1} {1.5*x1}\n")
            elif testNum == 2:
                x2 = 5*random.random()
                dataFile.write(f"{x1} {x2} {3*x1 + 2*x2 + 5}\n")
        return [fileName, str(inputSize), str(outputSize)]

def clear_data(test_num):
    user_input = input("Delete generated data files? [y/n] ")
    while user_input not in ['y', 'n']:
        user_input = input("Wrong input. [y/n] ")
    if user_input == 'y':
        for iter in range(1, test_num + 1):
            try:
                os.remove(f"data{iter}")
            except Exception as e:
                print(f"Failed to delete data{iter}. error: {e}")

cur_test = 1
# test 1
print("test 1")
args = generate_data(cur_test)
cmdLine = ["python3", "training_neural_network.py"] + args + consts
process = subprocess.run(cmdLine, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
print(f"error in test 1\n{process.stderr}") if process.returncode != 0 else print(process.stdout)

# test 2
cur_test += 1
print("test 2")
args = generate_data(cur_test)
cmdLine = ["python3", "training_neural_network.py"] + args + consts
process = subprocess.run(cmdLine, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
print(f"error in test 2\n{process.stderr}") if process.returncode != 0 else print(process.stdout)

clear_data(cur_test)