import numpy as np
from abc import ABC, abstractmethod
from gwu_nn.activation_layers import Sigmoid, RELU, Softmax

activation_functions = {'relu': RELU, 'sigmoid': Sigmoid, 'softmax': Softmax}


def apply_activation_forward(forward_pass):
    """Decorator that ensures that a layer's activation function is applied after the layer during forward
    propagation.
    """
    def wrapper(*args):
        output = forward_pass(args[0], args[1])
        if args[0].activation:
            return args[0].activation.forward_propagation(output)
        else:
            return output
    return wrapper


def apply_activation_backward(backward_pass):
    """Decorator that ensures that a layer's activation function's derivative is applied before the layer during
    backwards propagation.
    """
    def wrapper(*args):
        output_error = args[1]
        learning_rate = args[2]
        if args[0].activation:
            output_error = args[0].activation.backward_propagation(output_error, learning_rate)
        return backward_pass(args[0], output_error, learning_rate)
    return wrapper


class Layer():

    def __init__(self, activation=None):
        self.type = "Layer"
        if activation:
            self.activation = activation_functions[activation]()
        else:
            self.activation = None

    @apply_activation_forward
    def forward_propagation(cls, input):
        pass

    @apply_activation_backward
    def backward_propogation(cls, output_error, learning_rate):
        pass


class Dense(Layer):

    def __init__(self, output_size, add_bias=False, activation=None, input_size=None):
        super().__init__(activation)
        self.type = None
        self.name = "Dense"
        self.input_size = input_size
        self.output_size = output_size
        self.add_bias = add_bias


    def init_weights(self, input_size):
        """Initialize the weights for the layer based on input and output size

        Args:
            input_size (np.array): dimensions for the input array
        """
        if self.input_size is None:
            self.input_size = input_size

        self.weights = np.random.randn(input_size, self.output_size) / np.sqrt(input_size + self.output_size)
        if self.add_bias:
            self.bias = np.random.randn(1, self.output_size) / np.sqrt(input_size + self.output_size)


    @apply_activation_forward
    def forward_propagation(self, input):
        """Applies the forward propagation for a densely connected layer. This will compute the dot product between the
        input value (calculated during forward propagation) and the layer's weight tensor.

        Args:
            input (np.array): Input tensor calculated during forward propagation up to this layer.

        Returns:
            np.array(float): The dot product of the input and the layer's weight tensor."""
        self.input = input
        output = np.dot(input, self.weights)
        if self.add_bias:
            return output + self.bias
        else:
            return output

    @apply_activation_backward
    def backward_propagation(self, output_error, learning_rate):
        """Applies the backward propagation for a densely connected layer. This will calculate the output error
         (dot product of the output_error and the layer's weights) and will calculate the update gradient for the
         weights (dot product of the layer's input values and the output_error).

        Args:
            output_error (np.array): The gradient of the error up to this point in the network.

        Returns:
            np.array(float): The gradient of the error up to and including this layer."""
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        if self.add_bias:
            self.bias -= learning_rate * output_error
        return input_error

class Convolutional(Layer):

    def __init__(self, num_kernels=1, kernel_size=3, activation=None, input_size=None):
        super().__init__(activation)
        self.type = None
        self.name = "Convolutional"
        self.input = None # (img width, img height, input channels)
        self.kernels = None # (kernel_size, kernel_size, num_kernels)
        self.output = None # (img width, img height, num_kernels)

        self.input_size = input_size # (img width, img height, input channels)
        self.output_size = (kernel_size, kernel_size, num_kernels)
        self.kernel_size = kernel_size # (n, n) odd number
        self.num_kernels = num_kernels # corresponds to number of feature maps


    def init_weights(self, input_size):
        assert(len(input_size) == 3) # expects 3d ndarray
        """Initialize the weights for the layer based on input and output size

        Args:
            input_size (np.array): dimensions for the input array (expects 3d ndarray)
        """
        if self.input_size is None:
            self.input_size = input_size
        
        # initialize kernel weights (kernel_size, kernel_size, num_kernels)
        self.kernels = np.random.randn(self.kernel_size, self.kernel_size, self.num_kernels)


    @apply_activation_forward
    def forward_propagation(self, input):
        print(input.shape)
        assert(len(input.shape) == 3) # expects 3d ndarray
        """Applies the forward propagation for a convolutional layer. This will convolve the
        input value (calculated during forward propagation) with the layer's kernels.

        Args:
            input (np.array): Input tensor calculated during forward propagation up to this layer.

        Returns:
            np.array(float): An output tensor with shape (img_width, img_height,self.num_kernels)"""

        output = np.zeros(shape=(self.input_size[0], self.input_size[1], self.num_kernels))

        self.input = input
        input_pad = self.apply_2d_padding(input, self.kernel_size)

        for i_w in range(input.shape[0]): # img width
            for i_h in range(input.shape[1]): # img height
                for k_w in range(self.kernel_size):
                    for k_h in range(self.kernel_size):
                        for i in range(input.shape[2]): # input channels
                            for k in range(self.num_kernels): # output channels
                                output[i_w][i_h][k] += self.kernels[k_w][k_h][k] * input_pad[i_w+k_w][i_h+k_h][i]
                            
        return output


    @apply_activation_backward
    def backward_propagation(self, output_error, learning_rate):
        """Applies the backward propagation for a convolutional layer. This will calculate the output error
         (dot product of the output_error and the layer's weights) and will calculate the update gradient for the
         kernel weights (dot product of the layer's input values and the output_error).

        Args:
            output_error (np.array): The gradient of the error up to this point in the network.

        Returns:
            np.array(float): The gradient of the error up to and including this layer."""

        # calculate kernel gradient (need padded input)
        kernels_grad = np.zeros_like(self.kernels)
        input_pad = self.apply_2d_padding(self.input, self.kernel_size)
    
        # calculate input error (need padded output)
        input_error = np.zeros_like(self.input)    
        output_error_pad = self.apply_2d_padding(output_error, self.kernel_size)

        for i_w in range(self.input.shape[0]): # img width
            for i_h in range(self.input.shape[1]): # img height
                for k_w in range(self.kernel_size):
                    for k_h in range(self.kernel_size):
                        for i in range(self.input.shape[2]): # input channels
                            for k in range(self.num_kernels): # output channels
                                # calc kernel gradient and input_grad for i_w, i_h, k_w, k_h, i, k
                                kernels_grad[k_w][k_h][k] += input_pad[i_w+k_w][i_h+k_h][i] * output_error[k_w][k_h][k]
                                input_error[i_w][i_h][i] += output_error_pad[i_w+self.kernel_size-k_w-1][i_h+self.kernel_size-k_h-1][k] * self.kernels[k_w][k_h][k]
    
        # update kernel 
        self.kernels -= learning_rate * kernels_grad

        return input_error

    
    def apply_1d_padding(self, row, kernel_size):
        """ Helper function to pad 1d array with kernel_size//2 zeros on either side """
        padding = kernel_size//2
        channels = row.shape[-1]
        return np.concatenate([np.zeros(shape=(padding,channels)), row, np.zeros(shape=(padding,channels))])

    def apply_2d_padding(self, input_img, kernel_size):
        """ Helper function to apply 2d padding to a 3d array,
        pads with kernel_size//2 zeros on all sides """
        width = input_img.shape[1]
        channels = input_img.shape[2]
        padding = kernel_size//2

        pad_sides = np.stack([self.apply_1d_padding(row,kernel_size) for row in input_img])
        zeros = np.zeros(shape=(padding,width+2*padding,channels))
        pad_full = np.vstack([zeros, pad_sides, zeros])
        return pad_full


#Class Flatten(Layer):