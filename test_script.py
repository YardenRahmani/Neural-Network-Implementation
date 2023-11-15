import subprocess
import os
import random

LEARNING_RATE = 0.0001
HIDDEN_LAYER_SIZE = [8,4]
SAMPLES = 1000
EPOCHES = 10000
consts = [str(LEARNING_RATE), str(HIDDEN_LAYER_SIZE), str(EPOCHES)]

def generate_data(testNum):
    fileName = "data" + str(testNum)
    with open(fileName, "w") as dataFile:
        if testNum == 1 or testNum == 3:
            inputSize = 1
        elif testNum == 2 or testNum == 4:
            inputSize = 2
        outputSize = 1
        for _ in range(SAMPLES):
            x1 = 5*random.random() - 2.5
            if testNum == 1:
                dataFile.write(f"{x1} {1.5*x1}\n")
            elif testNum == 2:
                x2 = 5*random.random() - 2.5
                dataFile.write(f"{x1} {x2} {3*x1 + 2*x2 + 5}\n")
            elif testNum == 3:
                dataFile.write(f"{x1} {x1**2}\n")
            elif testNum == 4:
                x2 = 5*random.random() - 2.5
                dataFile.write(f"{x1} {x2} {5*x1+2*x2**2}\n")
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

for cur_test in range(1,5):
    print(f"test {cur_test}")
    args = generate_data(cur_test)
    cmdLine = ["python3", "training_neural_network.py"] + args + consts
    process = subprocess.run(cmdLine, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    print(f"error in test 1\n{process.stderr}") if process.returncode != 0 else print(process.stdout)
#print(process.stdout)
clear_data(cur_test)