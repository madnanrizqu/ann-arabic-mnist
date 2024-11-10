from network.network import NeuralNetwork
from network.layer import Layer
from network.activation import Activation
from helpers.activations import tanh, tanh_prime
from helpers.losses import mse, mse_prime
from helpers.dataset_prep import (
    convert_dataset_to_numpy,
    split_dataset,
    preprocess_dataset,
    distribute_class,
)
import numpy as np
from helpers.benchmark import calculate_accuracy

print("Dataset ready")

print("Preprocessing...")
x_train, y_train, x_test, y_test = preprocess_dataset(
    split_dataset(distribute_class(convert_dataset_to_numpy()))
)

# network
print("Setting up neural network...")
net = NeuralNetwork()
net.add(Layer(28 * 28, 100))  # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(Activation(tanh, tanh_prime))
net.add(Layer(100, 50))  # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(Activation(tanh, tanh_prime))
net.add(Layer(50, 10))  # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(Activation(tanh, tanh_prime))

# training
print("Training...")
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=100, learning_rate=0.001)

# benchmark
print("Testing...")
predictions = []
for i in range(len(x_test)):
    [all, single] = net.predict(x_test[i])
    predictions.append(all)

predictions = np.concatenate(predictions)
accuracy = calculate_accuracy(predictions, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
