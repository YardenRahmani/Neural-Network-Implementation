import sys
import numpy as np

def activation(x):
    return np.maximum(0, x)
    #return x

def activation_derivative(x):
    return np.where(x > 0, 1, 0)
    #return 1

def read_data(training_file_name, input_size, output_size):
    input_data = []
    output_data = []
    with open(training_file_name, "r") as dataFile:
        for line in dataFile.readlines():
            line = list(map(float,line.strip().split()))
            if len(line) != input_size + output_size:
                return None, None, False
            input_data.append(line[:input_size])
            output_data.append(line[input_size:])
    return np.array(input_data), np.array(output_data), True

def initialize_weights(layers_sizes):
    weights = []
    biases = []
    np.random.seed(0)
    for cur_layer in range(len(layers_sizes) - 1):
        weights.append(np.random.randn(layers_sizes[cur_layer],layers_sizes[cur_layer + 1]))
        biases.append(np.random.randn(1,layers_sizes[cur_layer + 1]))
    return weights, biases

def feedforward(X, weights, biases):
    layers_nodes_pre = []
    layer_nodes = [X]
    for layer in range(len(weights)):
        layers_nodes_pre.append(np.dot(layer_nodes[layer],weights[layer]) + biases[layer])
        layer_nodes.append(activation(layers_nodes_pre[layer]))
    return layers_nodes_pre, layer_nodes

def backpropagate(layers_nodes_pre_rev, layers_nodes_rev, weights_rev, error):
    ## X--(w0+b0)--G0--(f)--H0--(W1+b1)--G1--(f)-...-Y
    #print(f"MSE: {np.mean(error**2)}")
    samples = X.shape[0]
    grad_MSE_2_weights_rev = []
    grad_MSE_2_biases_rev = []
    grad_MSE_2_pre_layer_rev = []
    grad_MSE_2_layer_rev = [2*error] ## dL/dy = 2*error
    for layer in range(len(weights_rev)):
        ## dL/dGi = (dL/dHi)(dHi/dGi) = (dL/dHi)*relu_deriv(Gi)
        grad_MSE_2_pre_layer_rev.append(activation_derivative(layers_nodes_pre_rev[layer])*grad_MSE_2_layer_rev[layer])
        ## dL/dwi = (dL/dGi)(dGi/dwi) = (dL/dGi)*Hi-1
        grad_MSE_2_weights_rev.append(np.dot(layers_nodes_rev[layer+1].T, grad_MSE_2_pre_layer_rev[layer])/samples)
        ## dL/dbi = (dL/dGi)(dGi/dbi) = dL/dGi
        grad_MSE_2_biases_rev.append(np.mean(grad_MSE_2_pre_layer_rev[layer]))
        ## dL/dHi-1 = (dL/Gi)(dGi/dHi-1) = (dL/dGi)Wi
        grad_MSE_2_layer_rev.append(np.dot(grad_MSE_2_pre_layer_rev[layer],weights_rev[layer].T)/samples)
    return list(reversed(grad_MSE_2_weights_rev)), list(reversed(grad_MSE_2_biases_rev))

def gradient_descent(weights, biases, delta_w, delta_b, learning_rate):
    weights = [w - learning_rate * dw for w, dw in zip(weights, delta_w)]
    biases = [b - learning_rate * db for b, db in zip(biases, delta_b)]
    return weights, biases

def train_model(X, Y, learning_rate,  weights, biases, epochs, batches):
    for epoch in range(epochs):
        layers_nodes_pre, layer_nodes = feedforward(X, weights, biases)
        error = layer_nodes[-1] - Y
        MSE = np.mean(error**2)
        delta_w, delta_b = backpropagate(list(reversed(layers_nodes_pre)), list(reversed(layer_nodes)), list(reversed(weights)), error)
        weights, biases = gradient_descent(weights, biases, delta_w, delta_b, learning_rate)
    return MSE, weights, biases

if __name__ == "__main__":
    training_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    input_size = int(sys.argv[3])
    output_size = int(sys.argv[4])
    X, Y, valid = read_data(training_file_name, input_size, output_size)
    if valid == False:
        print("Bad training data file")
    else:
        epochs = int(sys.argv[5])
        batches = int(sys.argv[6])
        learning_rate = float(sys.argv[7])
        hidden_layers_sizes = list(map(int, list(sys.argv[8][1:-1].split(','))))
        weights, biases = initialize_weights([X.shape[1]] + hidden_layers_sizes + [Y.shape[1]])
        _, weights, biases = train_model(X, Y, learning_rate, weights, biases, epochs, batches)
        X_test, Y_test, valid = read_data(test_file_name, input_size, output_size)
        if valid == False:
            print("Bad testing data file")
        else:
            _, layer_nodes = feedforward(X_test, weights, biases)
            final_error = layer_nodes[-1] - Y_test
            print(f"Final error is {abs(np.mean(final_error))}")