import sys
import numpy as np

def read_data(file_name, input_size, output_size):
    input_data = []
    output_data = []
    with open(file_name, "r") as dataFile:
        for line in dataFile.readlines():
            line = list(map(float,line.strip().split()))
            if len(line) != input_size + output_size:
                return None, None, False
            input_data.append(line[:input_size])
            output_data.append(line[input_size:])
    return np.array(input_data), np.array(output_data), True

def initialize_weights(input_size, output_size, hidden_layer_size):
    np.random.seed(0)
    W1 = np.random.randn(input_size,hidden_layer_size)
    b1 = np.random.randn(1,hidden_layer_size)
    W2 = np.random.randn(hidden_layer_size,hidden_layer_size)
    b2 = np.random.randn(1, hidden_layer_size)
    W3 = np.random.randn(hidden_layer_size,output_size)
    b3 = np.random.randn(1, output_size)
    return W1, W2, W3, b1, b2, b3

def train_model(X, Y, learning_rate,  hidden_layer_size, samples, epochs):
    W1, W2, W3, b1, b2, b3 = initialize_weights(X.shape[1], Y.shape[1], hidden_layer_size)
    for epoch in range(epochs):
        H1 = np.dot(X,W1) + b1
        H2 = np.dot(H1,W2) + b2
        Y_pred = np.dot(H2,W3) + b3
        error = Y_pred - Y
        MSE = np.mean(error**2)
        #print(f"MSE: {MSE}")
        gradMSE2out = 2*error ## dL/dy = 2*error
        gradMSE2W3 = np.dot(H2.T, gradMSE2out) / samples ## dL/dw3 = (dL/dy)(dy/dW3) = (dL/dy)H2
        gradMSE2b3 = np.mean(gradMSE2out) ## dL/db3 = (dL/dy)(dy/b3) = dL/dy
        gradMSE2H2 = np.dot(gradMSE2out, W3.T) / samples ## dL/dH2 = (dL/dy)(dy/dH2) = (dL/dy)W3
        gradMSE2W2 = np.dot(H1.T, gradMSE2H2) / samples## dL/dW2 = (dL/dH2)(dH2/dW2) = (dL/dH2)H1
        gradMSE2b2 = np.mean(gradMSE2H2) ## dL/db2 = (dL/dH2)(dH2/db2) = dL/dH2
        gradMSE2H1 = np.dot(gradMSE2H2, W2.T) / samples ## dL/dH1 = (dL/dH2)(dH2/dH1) = (dL/dH2)W2
        gradMSE2W1 = np.dot(X.T, gradMSE2H1) / samples ## dL/dW1 = (dL/dH1)(dH1/dW1) = (dL/dH)X
        gradMSE2b1 = np.mean(gradMSE2H1) ## dL/db1 = (dL/dH)(dH/db1) = dL/dH
        W3 -= learning_rate * gradMSE2W3
        b3 -= learning_rate * gradMSE2b3
        W2 -= learning_rate * gradMSE2W2
        b2 -= learning_rate * gradMSE2b2
        W1 -= learning_rate * gradMSE2W1
        b1 -= learning_rate * gradMSE2b1
        if MSE < 1e-05:
            print(f"after {epoch + 1} iterations: MSE = {MSE}")
            break

if __name__ == "__main__":
    file_name = sys.argv[1]
    input_size = int(sys.argv[2])
    output_size = int(sys.argv[3])
    X, Y, valid = read_data(file_name, input_size, output_size)
    if valid == False:
        print("Bad data file")
    else:
        learning_rate = float(sys.argv[4])
        hidden_layer_size = int(sys.argv[5])
        samples = int(sys.argv[6])
        epochs = int(sys.argv[7])
        print(X.shape[1])
        train_model(X, Y, learning_rate, hidden_layer_size, samples, epochs)