from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime

from keras.datasets import mnist
from keras.utils import to_categorical

from benchmark import calculate_accuracy
import numpy as np

# load MNIST from server
print("Downloading dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
print("Preprocessing...")
x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28)
x_train = x_train.astype("float32")
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = to_categorical(y_train)

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28)
x_test = x_test.astype("float32")
x_test /= 255
y_test = to_categorical(y_test)

# network
print("Setting up neural network...")
net = Network()
net.add(FCLayer(28 * 28, 100))  # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100, 50))  # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50, 10))  # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(ActivationLayer(tanh, tanh_prime))

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
