import subprocess
import os
import random
from itertools import product

LEARNING_RATES = [pow(10, -x) for x in reversed(range(1,6))]
SIZES_RANGE = range(4,13,8)
HIDDEN_LAYER_SIZES = [[x] for x in SIZES_RANGE] + list(product(SIZES_RANGE,SIZES_RANGE))
HIDDEN_LAYER_SIZES += list(product(SIZES_RANGE,SIZES_RANGE,SIZES_RANGE))
HIDDEN_LAYER_SIZES += list(product(SIZES_RANGE,SIZES_RANGE,SIZES_RANGE,SIZES_RANGE))
SAMPLES = 1000
SAMPLES_RANGE = 100
EPOCHES = 100
BATCHES = 1

def generate_data(testNum):
    fileName = "data" + str(testNum)
    with open(fileName, "w") as dataFile:
        if testNum == 1 or testNum == 3:
            inputSize = 1
        elif testNum == 2 or testNum == 4:
            inputSize = 2
        outputSize = 1
        for _ in range(SAMPLES):
            x1 = SAMPLES_RANGE*(random.random() - 0.5)
            if testNum == 1:
                dataFile.write(f"{x1} {1.5*x1}\n")
            elif testNum == 2:
                x2 = SAMPLES_RANGE*(random.random() - 0.5)
                dataFile.write(f"{x1} {x2} {3*x1 + 2*x2 + 5}\n")
            elif testNum == 3:
                dataFile.write(f"{x1} {x1**2}\n")
            elif testNum == 4:
                x2 = SAMPLES_RANGE*(random.random() - 0.5)
                dataFile.write(f"{x1} {x2} {5*x1+2*x2**2}\n")
        return [fileName, inputSize, outputSize]

def clear_data(test_num):
    user_input = input("\nDelete generated data files? [y/n] ")
    while user_input not in ['y', 'n']:
        user_input = input("Wrong input. [y/n] ")
    if user_input == 'y':
        for iter in range(1, test_num + 1):
            try:
                os.remove(f"data{iter}")
            except Exception as e:
                print(f"Failed to delete data{iter}. error: {e}")
    user_input = input("Delete log files? [y/n] ")
    while user_input not in ['y', 'n']:
        user_input = input("Wrong input. [y/n] ")
    if user_input == 'y':
        for iter in range(1, test_num + 1):
            try:
                os.remove(f"log{iter}.txt")
            except Exception as e:
                print(f"Failed to delete log{iter}. error: {e}")

for cur_test in range(1,5):
    print(f"test {cur_test}")
    with open(f"log{cur_test}.txt", 'w') as logFile:
        logFile.write(f"test {cur_test}\n")
    training_set, inputSize, outputSize = generate_data(cur_test)
    #validation_set, _, _ = generate_data(cur_test)
    test_set, _, _ = generate_data(cur_test)
    cmd_line = ["python3", "training_neural_network.py", training_set, test_set, str(inputSize), str(outputSize), str(EPOCHES), str(BATCHES)]
    min_error = float('inf')
    for learning_rate in LEARNING_RATES:
        for layers in HIDDEN_LAYER_SIZES:
            cur_cmd = cmd_line + [str(learning_rate), str(layers)]
            process = subprocess.run(cur_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            if process.returncode != 0:
                print(f"error in test {cur_test}\n{process.stderr}")
                exit()
            else:
                cur_error = float(process.stdout.split()[-1])
                if cur_error < min_error:
                    best_rate = learning_rate
                    best_config = layers
                    min_error = cur_error
                with open(f"log{cur_test}.txt", 'a+') as logFile:
                    logFile.write(f"Learning rate: {learning_rate}, layers: {layers}, error: {cur_error}, best: {min_error}\n")
    error_precent = 100*min_error/(SAMPLES_RANGE/2.0)
    print(f"With learning rate of {best_rate} and layers size {best_config}, the mean error is {error_precent:.3g}%") if min_error != float('inf') else None

clear_data(cur_test)