import numpy as np

from network.network import NeuralNetwork
from network.layer import Layer
from network.activation import Activation
from helpers.activations import tanh, tanh_prime
from helpers.losses import mse, mse_prime

np.random.seed(5)

# training data
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

x_test = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_test = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = NeuralNetwork()
net.add(Layer(2, 3))
net.add(Activation(tanh, tanh_prime))
net.add(Layer(3, 1))
net.add(Activation(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=10000, learning_rate=0.01)

# benchmark
predictions = []
for i in range(len(x_test)):
    [out, single] = net.predict(x_test[i])
    predictions.append(out)

print(predictions)
print(y_test)
