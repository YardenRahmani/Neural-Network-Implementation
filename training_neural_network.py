import sys
import numpy as np

def read_data():
    input = []
    output = [] 
    with open(sys.argv[1], "r") as dataFile:
        for line in dataFile.readlines():
            line = list(map(float,line.strip().split()))
            input.append(line[:inputSize])
            output.append(line[inputSize:])
    return np.array(input), np.array(output)

def train_model():
    LEARNING_RATE = float(sys.argv[4])
    HIDDEN_LAYER_SIZE = int(sys.argv[5])
    SAMPLES = int(sys.argv[6])
    EPOCHES = int(sys.argv[7])
    np.random.seed(0)
    W1 = np.random.randn(inputSize,HIDDEN_LAYER_SIZE)
    b1 = np.random.randn(1,HIDDEN_LAYER_SIZE)
    W2 = np.random.randn(HIDDEN_LAYER_SIZE,HIDDEN_LAYER_SIZE)
    b2 = np.random.randn(1, HIDDEN_LAYER_SIZE)
    W3 = np.random.randn(HIDDEN_LAYER_SIZE,outputSize)
    b3 = np.random.randn(1, outputSize)
    for epoch in range(EPOCHES):
        H1 = np.dot(X,W1) + b1
        H2 = np.dot(H1,W2) + b2
        Y_pred = np.dot(H2,W3) + b3
        error = Y_pred - Y
        MSE = np.mean(error**2)
        #print(f"MSE: {MSE}")
        gradMSE2out = 2*error ## dL/dy = 2*error
        gradMSE2W3 = np.dot(H2.T, gradMSE2out) / SAMPLES ## dL/dw3 = (dL/dy)(dy/dW3) = (dL/dy)H2
        gradMSE2b3 = np.mean(gradMSE2out) ## dL/db3 = (dL/dy)(dy/b3) = dL/dy
        gradMSE2H2 = np.dot(gradMSE2out, W3.T) / SAMPLES ## dL/dH2 = (dL/dy)(dy/dH2) = (dL/dy)W3
        gradMSE2W2 = np.dot(H1.T, gradMSE2H2) / SAMPLES## dL/dW2 = (dL/dH2)(dH2/dW2) = (dL/dH2)H1
        gradMSE2b2 = np.mean(gradMSE2H2) ## dL/db2 = (dL/dH2)(dH2/db2) = dL/dH2
        gradMSE2H1 = np.dot(gradMSE2H2, W2.T) / SAMPLES ## dL/dH1 = (dL/dH2)(dH2/dH1) = (dL/dH2)W2
        gradMSE2W1 = np.dot(X.T, gradMSE2H1) / SAMPLES ## dL/dW1 = (dL/dH1)(dH1/dW1) = (dL/dH)X
        gradMSE2b1 = np.mean(gradMSE2H1) ## dL/db1 = (dL/dH)(dH/db1) = dL/dH
        W3 -= LEARNING_RATE * gradMSE2W3
        b3 -= LEARNING_RATE * gradMSE2b3
        W2 -= LEARNING_RATE * gradMSE2W2
        b2 -= LEARNING_RATE * gradMSE2b2
        W1 -= LEARNING_RATE * gradMSE2W1
        b1 -= LEARNING_RATE * gradMSE2b1
        if MSE < 1e-05:
            print(f"after {epoch + 1} iterations: MSE = {MSE}")
            break

inputSize = int(sys.argv[2])
outputSize = int(sys.argv[3])
X, Y = read_data()
train_model()