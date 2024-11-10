from network.base_layer import BaseLayer


class Activation(BaseLayer):
    """
    applies an activation function. acts as an layer
    """

    def __init__(self, activation_function, activation_derivative):
        """
        initialize activation layer

        args:
            activation_function: the activation function to apply (e.g., ReLU, sigmoid)
            activation_derivative: derivative of the activation function for backprop
        """
        super().__init__()  # Initialize parent class
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative

    def forward_propagation(self, input_data):
        """
        apply activation function to input data

        args:
            input_data: input values to activate

        returns:
            activated values
        """
        # store input for use in backprop
        self.input_data = input_data
        # apply activation function
        self.output_data = self.activation_function(self.input_data)
        return self.output_data

    def backward_propagation(self, output_gradient, learning_rate):
        """
        compute gradient for backpropagation

        args:
            output_gradient: gradient from next layer
            learning_rate: not used since activation has no learnable parameters

        returns:
            gradient with respect to input
        """
        # chain rule: multiply output gradient by activation derivative
        return self.activation_derivative(self.input_data) * output_gradient
