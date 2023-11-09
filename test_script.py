import subprocess
import random

LEARNING_RATE = 0.001
HIDDEN_LAYER_SIZE = 5
SAMPLES = 1000
EPOCHES = 10000
consts = [str(LEARNING_RATE), str(HIDDEN_LAYER_SIZE), str(SAMPLES), str(EPOCHES)]

def generate_data(testNum):
    fileName = "data" + str(testNum)
    with open(fileName, "w") as dataFile:
        if testNum == 1:
            inputSize = 1
        elif testNum == 2:
            inputSize = 2
        outputSize = 1
        for i in range(SAMPLES):
            x1 = 5*random.random()
            if testNum == 1:
                dataFile.write(f"{x1} {1.5*x1}\n")
            elif testNum == 2:
                x2 = 5*random.random()
                dataFile.write(f"{x1} {x2} {3*x1 + 2*x2 + 5}\n")
        return [fileName, str(inputSize), str(outputSize)]

# test 1
print("test 1")
args = generate_data(1)
cmdLine = ["python3", "training_neural_network.py"] + args + consts
process = subprocess.run(cmdLine, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
if process.returncode != 0:
    print(f"error in test 1")
print(process.stdout)

# test 2
print("test 2")
args = generate_data(2)
cmdLine = ["python3", "training_neural_network.py"] + args + consts
process = subprocess.run(cmdLine, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
if process.returncode != 0:
    print(f"error in test 2")
print(process.stdout)