import numpy as np

class Layer():

    def __init__(self, layer_config,network_l_rate,network_wreg) -> None:

        # ------------------ Dictionnary Activation Function ------------------
        activation_functions = {
                "sigmoid": self.sigmoid,
                "relu": self.relu,
                "tanh": self.tanh,
                "softmax": self.softmax,
                "linear": self.linear
                }
        
        # ------------------ Dictionnary Activation Derivated Function ------------------
        activation_functions_der = {
                "sigmoid": self.sigmoid_der,
                "relu": self.relu_der,
                "tanh": self.tanh_der,
                "softmax": None,
                "linear": self.linear_der
                }

        # ------------------ Load the parameters ------------------
        self.type = layer_config.get("type", "basic_layer")                      # Type of the layer

        if self.type in ["basic_layer"]:
            self.size = layer_config.get("size", 1)                              # Size of the layer
            self.w = None                                                        # Weights
            self.bias = None                                                     # Bias
            self.act = activation_functions[layer_config.get("act", "sigmoid")]  # Activation function
            self.der_act = activation_functions_der[layer_config.get("act", "sigmoid")]  # Activation derivated function
            self.wr = layer_config.get("wr", [-0.1, 0.1])                        # Weight range
            self.lrate = layer_config.get("lrate", network_l_rate)                         # Learning rate
            self.br = layer_config.get("br", [0,1])                              # Bias range
            self.wreg = layer_config.get("wreg", network_wreg)                              # Weight regularization

        elif self.type == "softmax":
            self.w = layer_config.get("size", None)                              # Size of the layer
            self.act = activation_functions["softmax"]                           # Activation function

        # ------------------  Atributes  ------------------
        self.input = None           # Input of the layer
        self.output = None          # Output of the layer
        self.grad_w = None          # Gradient of the weights
        self.grad_bias = None       # Gradient of the bias

    # -------------- List of activation functions --------------
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    
    def relu(self, X):
        return np.maximum(0, X)
    
    def tanh(self, X):
        return np.tanh(X)
    
    def softmax(self, X):
        exps = np.exp(X - np.max(X))
        return exps / np.sum(exps)
    
    def linear(self, X):
        return X
    
    # -------------- List of activation derivated functions --------------
    def sigmoid_der(self, X):
        return self.sigmoid(X) * (1 - self.sigmoid(X))
    
    def relu_der(self, X):
        return np.where(X <= 0, 0, 1)
    
    def tanh_der(self, X):
        return 1 - np.tanh(X)**2

    def linear_der(self, X):
        return 1
    

    # -------------- Initialize the weights --------------
    def init_weights(self,len_input):
        self.w = np.random.uniform(self.wr[0], self.wr[1], (len_input,self.size))   # verify the size !!!!
        self.bias = np.random.uniform(self.br[0], self.br[1],self.size)


    # -------------- Forward function --------------
    def forward(self, input):
        self.input = input    # Save the input
        if self.type != "softmax":
            self.output = np.dot(input,self.w) + self.bias
        else: self.output = input
        self.output = self.act(self.output)

    # -------------- Backward function --------------
    def backward(self, g, wrt_function):
        """
        g is the gradient of the last layer's output
        g : shape (number cases, number of classes)
        """

        if self.type != "softmax":
            # Converte the gradient of the layer's output into a gradient on the prenonlinearity activation
            # NOTE: it's Hadamard product
            g = g * self.der_act(self.output)

            # Gradient of weights and bias
            self.grad_w = np.mean(np.einsum('ij,ik->ijk', self.input, g), axis=0) + self.wreg * wrt_function(self.w)
            self.grad_bias = g.mean(axis=0)

            # Propagate the gradient to the next layer
            g = np.dot(g, self.w.T)

        else:
            pass
        
        return g

    # -------------- Update the weights --------------
    def update(self):
        self.w -= self.lrate * self.grad_w
        self.bias -= self.lrate * self.grad_bias

    