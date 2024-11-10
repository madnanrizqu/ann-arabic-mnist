from network.network import Network
from network.layer import Layer
from network.activation import Activation
from helpers.activations import tanh, tanh_prime
from helpers.losses import mse, mse_prime

from keras.datasets import mnist
from keras.utils import to_categorical

from helpers.benchmark import calculate_accuracy
import numpy as np

# load MNIST from server
print("Downloading dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Preprocessing...")
x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28)
x_train = x_train.astype("float32")
x_train /= 255
y_train = to_categorical(y_train)

x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28)
x_test = x_test.astype("float32")
x_test /= 255
y_test = to_categorical(y_test)

# network
print("Setting up neural network...")
net = Network()
net.add(Layer(28 * 28, 100))
net.add(Activation(tanh, tanh_prime))
net.add(Layer(100, 50))
net.add(Activation(tanh, tanh_prime))
net.add(Layer(50, 10))
net.add(Activation(tanh, tanh_prime))

# training
print("Training...")
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=5, learning_rate=0.1)

# benchmark
print("Testing...")
predictions = []
for i in range(len(x_test)):
    [all, single] = net.predict(x_test[i])
    predictions.append(all)

predictions = np.concatenate(predictions)

# print(predictions)
# calculate and print accuracy
accuracy = calculate_accuracy(predictions, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
