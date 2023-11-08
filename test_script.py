import subprocess
import random

LEARNING_RATE = 0.01
HIDDEN_LAYER_SIZE = 7
SAMPLES = 1000
EPOCHES = 1000

def generate_data(testNum):
    fileName = "data" + str(testNum)
    inputSize = 2
    outputSize = 1
    with open(fileName, "w") as dataFile:
        for i in range(SAMPLES):
            x1 = 5*random.random()
            x2 = 5*random.random()
            dataFile.write(f"{x1} {x2} {3*x1 + 2*x2 + 5}\n")
        return [fileName, str(inputSize), str(outputSize)]

# test 1
print("test 1")
args = generate_data(1)
consts = [str(LEARNING_RATE), str(HIDDEN_LAYER_SIZE), str(SAMPLES), str(EPOCHES)]
cmdLine = ["python3", "training_neural_network.py"] + args + consts
process = subprocess.run(cmdLine, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
if process.returncode != 0:
    print(f"error in test 1")
    #exit
print(process.stdout)