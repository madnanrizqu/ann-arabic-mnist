import numpy as np

from network import Network
from layer import FCLayer
from activation import Activation
from activations import tanh, tanh_prime
from losses import mse, mse_prime

# training data
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

x_test = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_test = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(FCLayer(2, 3))
net.add(Activation(tanh, tanh_prime))
net.add(FCLayer(3, 1))
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
