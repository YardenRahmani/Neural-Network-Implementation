import numpy as np

np.random.seed(0)
LEARNING_RATE = 0.01
HIDDEN_LAYER_SIZE = 7
SAMPLES = 1000
EPOCHES = 1000
X = 5 * np.random.randn(SAMPLES, 2)
Y = np.array([X[:,0]*3 + X[:,1]*2] + np.ones((1,SAMPLES))).T
W1 = np.random.randn(2,HIDDEN_LAYER_SIZE)
b1 = np.random.randn(1,HIDDEN_LAYER_SIZE)
W2 = np.random.randn(HIDDEN_LAYER_SIZE,1)
b2 = np.random.randn(1)

for epoch in range(EPOCHES):
    H = np.dot(X,W1) + b1
    Y_pred = np.dot(H,W2) + b2
    error = Y_pred - Y
    MSE = np.mean(error**2)
    gradMSE2out = 2*error ## dL/dy = 2*error
    print(MSE)
    gradMSE2W2 = np.dot(H.T, gradMSE2out) / SAMPLES ## dL/dw2 = (dL/dy)(dy/dW2) = (dL/dy)H
    gradMSE2b2 = np.mean(gradMSE2out) ## dL/db2 = (dL/dy)(dy/b2) = dL/dy
    gradMSE2H = np.dot(gradMSE2out, W2.T) / SAMPLES ## dL/dH = (dL/dy)(dy/dH) = (dL/dy)W2
    dW1 = np.dot(X.T, gradMSE2H) / SAMPLES ## dL/dW1 = (dL/dH)(dH/dW1) = (dL/dH)X
    db1 = np.mean(gradMSE2H) ## dL/db1 = (dL/dH)(dH/db1) = dL/dH
    W2 -= LEARNING_RATE * gradMSE2W2
    b2 -= LEARNING_RATE * gradMSE2b2
    W1 -= LEARNING_RATE * dW1
    b1 -= LEARNING_RATE * db1
    if MSE < 1e-07:
        print(f"after {epoch} iterations")
        break