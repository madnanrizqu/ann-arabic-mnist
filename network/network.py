import numpy as np


class NeuralNetwork:
    """
    neural network implementation supporting multiple layers and custom loss functions
    """

    def __init__(self):
        """initialize an empty neural network"""
        self.layers = []  # list to store network layers
        self.loss_function = None  # loss function for training
        self.loss_derivative = None  # derivative of loss function for backprop

    def add(self, layer):
        """
        add a layer to the network

        args:
            layer: Layer object to add to the network
        """
        self.layers.append(layer)

    def use(self, loss_function, loss_derivative):
        """
        set the loss function for training

        args:
            loss_function: function to calculate loss
            loss_derivative: derivative of loss function for backprop
        """
        self.loss_function = loss_function
        self.loss_derivative = loss_derivative

    def predict(self, input_data):
        """
        generate predictions for input data

        args:
            input_data: input samples to predict

        returns:
            list containing [all_layer_outputs, predicted_classes]
        """
        num_samples = len(input_data)
        layer_outputs = []  # store full layer outputs
        predicted_classes = []  # store single class predictions

        # process each sample
        for i in range(num_samples):
            # forward propagation through all layers
            current_output = input_data[i]
            for layer in self.layers:
                current_output = layer.forward_propagation(current_output)

            # store results
            layer_outputs.append(current_output)
            predicted_classes.append(np.argmax(current_output, axis=1)[0])

        return [layer_outputs, predicted_classes]

    def fit(self, x_train, y_train, epochs, learning_rate):
        """
        train the neural network

        args:
            x_train: training input data
            y_train: training target data
            epochs: number of training epochs
            learning_rate: learning rate for gradient descent
        """
        num_samples = len(x_train)

        # training loop
        for epoch in range(epochs):
            total_error = 0

            # process each training sample
            for i in range(num_samples):
                # forward propagation
                current_output = x_train[i]
                for layer in self.layers:
                    current_output = layer.forward_propagation(current_output)

                # calculate error
                total_error += self.loss_function(y_train[i], current_output)

                # backward propagation
                gradient = self.loss_derivative(y_train[i], current_output)
                for layer in reversed(self.layers):
                    gradient = layer.backward_propagation(gradient, learning_rate)

            # calculate average error
            average_error = total_error / num_samples
            print(f"Epoch {epoch + 1}/{epochs} - Average Error: {average_error:.6f}")
