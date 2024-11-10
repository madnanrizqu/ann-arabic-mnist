from base_layer import BaseLayer
import numpy as np


class Layer(BaseLayer):
    """
    fully connected (dense) neural network layer
    """

    def __init__(self, input_neurons, output_neurons):
        """
        initialize fully connected layer

        args:
            input_neurons: number of input features
            output_neurons: number of neurons in this layer
        """
        super().__init__()
        # Initialize weights with small random values centered around 0
        self.weights = np.random.rand(input_neurons, output_neurons) - 0.5
        # Initialize bias terms
        self.bias = np.random.rand(1, output_neurons) - 0.5

    def forward_propagation(self, input_data):
        """
        compute layer outputs: output = input @ weights + bias

        args:
            input_data: input features (batch_size, input_neurons)

        Returns:
            Layer output (batch_size, output_neurons)
        """
        self.input_data = input_data

        # linear transformation: y = Wx + b
        self.output_data = np.dot(self.input_data, self.weights) + self.bias
        return self.output_data

    def backward_propagation(self, output_gradient, learning_rate):
        """
        update weights and biases using gradient descent

        args:
            output_gradient: gradient from next layer
            learning_rate: step size for parameter updates

        returns:
            gradient with respect to input
        """

        # compute gradients
        input_gradient = np.dot(output_gradient, self.weights.T)  # dE/dX
        weights_gradient = np.dot(self.input_data.T, output_gradient)  # dE/dW

        # bias_gradient = output_gradient (directly) # dE/dB

        # update parameters using gradient descent
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient

        return input_gradient
