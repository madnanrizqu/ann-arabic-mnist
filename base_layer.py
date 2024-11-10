class BaseLayer:
    """
    abstract base class for neural network layers
    """

    def __init__(self):
        # store input and output values for each layer
        self.input_data = None
        self.output_data = None

    def forward_propagation(self, input_data):
        """
        process input data and compute layer output

        args:
            input_data: input values to process

        returns:
            layer output

        raises:
            NotImplementedError: must be implemented by child classes
        """
        raise NotImplementedError("Forward propagation not implemented!")

    def backward_propagation(self, error_gradient, learning_rate):
        """
        compute gradients and update layer parameters

        args:
            error_gradient: gradient of error with respect to layer output
            learning_rate: step size for parameter updates

        returns:
            gradient of error with respect to layer input

        raises:
            NotImplementedError: must be implemented by child classes
        """
        raise NotImplementedError("Backward propagation not implemented!")
