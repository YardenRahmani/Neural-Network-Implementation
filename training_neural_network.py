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

def initialize_weights(layers):
    weights = []
    biases = []
    np.random.seed(0)
    for cur_layer in range(len(layers) - 1):
        weights.append(np.random.randn(layers[cur_layer],layers[cur_layer + 1]))
        biases.append(np.random.randn(1,layers[cur_layer + 1]))
    return weights, biases

def feedforward(weights, biases):
    H1 = np.dot(X,weights[0]) + biases[0]
    H2 = np.dot(H1,weights[1]) + biases[1]
    Y_pred = np.dot(H2,weights[2]) + biases[2]
    return H1, H2, Y_pred

def backpropagate(X, H1, H2, weights, error):
    samples = X.shape[0]
    #print(f"MSE: {MSE}")
    grad_MSE_2_out = 2*error ## dL/dy = 2*error
    grad_MSE_2_W3 = np.dot(H2.T, grad_MSE_2_out) / samples ## dL/dw3 = (dL/dy)(dy/dW3) = (dL/dy)H2
    grad_MSE_2_b3 = np.mean(grad_MSE_2_out) ## dL/db3 = (dL/dy)(dy/b3) = dL/dy
    grad_MSE_2_H2 = np.dot(grad_MSE_2_out, weights[2].T) / samples ## dL/dH2 = (dL/dy)(dy/dH2) = (dL/dy)W3
    grad_MSE_2_W2 = np.dot(H1.T, grad_MSE_2_H2) / samples## dL/dW2 = (dL/dH2)(dH2/dW2) = (dL/dH2)H1
    grad_MSE_2_b2 = np.mean(grad_MSE_2_H2) ## dL/db2 = (dL/dH2)(dH2/db2) = dL/dH2
    grad_MSE_2_H1 = np.dot(grad_MSE_2_H2, weights[1].T) / samples ## dL/dH1 = (dL/dH2)(dH2/dH1) = (dL/dH2)W2
    grad_MSE_2_W1 = np.dot(X.T, grad_MSE_2_H1) / samples ## dL/dW1 = (dL/dH1)(dH1/dW1) = (dL/dH)X
    grad_MSE_2_b1 = np.mean(grad_MSE_2_H1) ## dL/db1 = (dL/dH)(dH/db1) = dL/dH
    return [grad_MSE_2_W1, grad_MSE_2_W2, grad_MSE_2_W3], [grad_MSE_2_b1, grad_MSE_2_b2, grad_MSE_2_b3]

def gradient_descent(weights, biases, delta_w, delta_b, learning_rate):
    weights[0] -= learning_rate * delta_w[0]
    weights[1] -= learning_rate * delta_w[1]
    weights[2] -= learning_rate * delta_w[2]
    biases[0] -= learning_rate * delta_b[0]
    biases[1] -= learning_rate * delta_b[1]
    biases[2] -= learning_rate * delta_b[2]
    return weights, biases

def train_model(X, Y, learning_rate,  hidden_layers, epochs):
    weights, biases = initialize_weights([X.shape[1]] + hidden_layers + [Y.shape[1]])
    for epoch in range(epochs):
        H1, H2, Y_pred = feedforward(weights, biases)
        error = Y_pred - Y
        MSE = np.mean(error**2)
        if MSE < 1e-05:
            print(f"after {epoch + 1} iterations: MSE = {MSE}")
            break
        delta_w, delta_b = backpropagate(X, H1, H2, weights, error)
        weights, biases = gradient_descent(weights, biases, delta_w, delta_b, learning_rate)

if __name__ == "__main__":
    file_name = sys.argv[1]
    input_size = int(sys.argv[2])
    output_size = int(sys.argv[3])
    X, Y, valid = read_data(file_name, input_size, output_size)
    if valid == False:
        print("Bad data file")
    else:
        learning_rate = float(sys.argv[4])
        hidden_layers = list(map(int, list(sys.argv[5][1:-1].split(','))))
        epochs = int(sys.argv[6])
        train_model(X, Y, learning_rate, hidden_layers, epochs)