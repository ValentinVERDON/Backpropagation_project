import Layer
import Data_generator
import numpy as np
import matplotlib.pyplot as plt

class Network():
    
    def __init__(self,config) -> None:

        global_config = config["GLOBALS"]
        layers_config = config["LAYERS"]

        # ------------------ Dictionnary Loss Function ------------------
        loss_functions = {
            "MSE": self.MSE,
            "cross_entropy": self.CE
        }

        # ------------------ Dictionnary Jacobian Loss Function ------------------
        jacobian_loss_functions = {
            "MSE": self.jacob_MSE,
            "cross_entropy": self.jacob_CE
        }

        # ------------------ Dictionnary Regularization Function ------------------
        reg_function = {
            "L1": lambda w: np.sign(w),
            "L2": lambda w: w
        }

        # ------------------ Load the data ------------------
        # Load the data with flatten images
        data = Data_generator.load_data(global_config["data"],flat=True)

        self.tr_data = data[0]                  # Training data
        self.test_data = data[1]                # Testing data
        self.val_data = data[2]                 # Validation data

        self.X_train = self.tr_data[0]                      # Training data
        self.y_train = np.array(self.tr_data[1])            # Training target
        self.X_test = self.test_data[0]                     # Testing data
        self.y_test = np.array(self.test_data[1])           # Testing target
        self.X_val = self.val_data[0]                       # Validation data
        self.y_val = np.array(self.val_data[1])             # Validation target

        # ------------------ Load the parameters ------------------
        # Load the general parameters
        self.loss_function = loss_functions[global_config["loss"]]              # Loss function
        self.jacobian_loss_function = jacobian_loss_functions[global_config["loss"]]              # Loss function jacobian
        self.lrate = global_config["lrate"]             # Learning rate
        self.wreg = global_config["wreg"]               # Weight regularization
        self.wrt_function = reg_function[global_config["wrt"]]                 # Weight regularization type

        # Load the layers
        self.layers = []
        for layer_config in layers_config:
            self.layers.append(Layer.Layer(layer_config, self.lrate, self.wreg))

            # Correct the input size of the softmax layer
            if self.layers[-1].type == "softmax":
                self.layers[-1].w = self.layers[-2].w

        # ------------------ Output parameters ------------------
        self.output = None
        self.loss = None

        # ------------------ Saving values ------------------
        self.train_loss = []
        self.test_loss = []
        self.val_loss = []
    
    # ------------------ Loss functions ------------------
    def MSE(self, output, target):
        return np.sum((output - target)**2)/output.shape[0]
        
    def CE(self, output, target):
        eps = 1e-15 # To avoid log(0)
        return -np.sum(target*np.log(output+eps))/output.shape[0]
        
    # ------------------ Initialize the weights ------------------
    def init_weights(self):
        # First Layer
        self.layers[0].init_weights(self.X_train.shape[1])
        # Other layers
        for index in range(1,len(self.layers)):
            if self.layers[index].type != "softmax":
                self.layers[index].init_weights(self.layers[index-1].size)
        
            
    # ------------------ Forward and backward ------------------
    def forward(self, input, target, verbose=False):
        """
        input : shape (number of cases, number of features)
        target : shape (number of cases, number of classes)
        """
        # First layer
        self.layers[0].forward(input)
        # Other layers
        for i in range(1,len(self.layers)):
            self.layers[i].forward(self.layers[i-1].output)
        # Set the output
        self.output = self.layers[-1].output
        
        # Calculate the loss
        self.loss = self.loss_function(self.output, target)

        # verbose:
        if verbose:
            print("Input : ",input)
            print("Output : ",self.output)
            print("Target : ",target)
            print("Loss : ",self.loss)


    def backward(self, target):
        """
        target : shape (number of cases, number of classes)
        """
        # Calculate the gradient of the loss function
        """ 
        jacob_loss : shape (number of cases, number of classes)
        """
        jacob_loss = self.jacobian_loss_function(self.output, target)

        """
        We follow the algorithm 6.4 from the book 
        """
        # We start from the last layer and go backward
        g =  jacob_loss
        for i in range(len(self.layers)-1,-1,-1):
            g = self.layers[i].backward(g,self.wrt_function)

    # ------------------ Update the weights ------------------
    def update(self):
        for i in range(len(self.layers)):
            if self.layers[i].type != "softmax":
                self.layers[i].update()

    # ------------------ Train the network ------------------
    def train(self, epochs=10):
        print("Training the network")
        for epoch in range(epochs):
            self.forward(self.X_train, self.y_train)
            self.backward(self.y_train)
            self.update()
            print("Epoch : ",epoch," Train Loss : ",self.loss)

            # Save the loss
            self.train_loss.append(self.loss)
            self.test_loss.append(self.loss_function(self.predict(self.X_val), self.y_val))
            

    # ------------------ Useful functions ------------------    
    def jacob_MSE(self, output, target):
        """
        output : shape (number cases,number of classes)
        target : shape (number cases, number of classes)
        """
        return 2*(output-target)
    
    def jacob_CE(self, output, target):
        """
        output : shape (number cases,number of classes)
        target : shape (number cases, number of classes)
        """
        eps = 1e-15 # To avoid log(0)
        return -target/(output+eps)

    # ------------------ Predict the output ------------------
    def predict(self, input):
        # First layer
        self.layers[0].forward(input)
        # Other layers
        for i in range(1,len(self.layers)):
            self.layers[i].forward(self.layers[i-1].output)
        # Set the output
        return self.layers[-1].output
    
    # ----------------- Test Score ------------------
    def test_score(self):
        return self.loss_function(self.predict(self.X_test), self.y_test)

    # ------------------ Visualize the results ------------------
    def visualize(self,zoom=False,factor=1):
        plt.plot(self.train_loss, label="Train Loss")
        plt.plot(self.test_loss, label="Valid Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        if zoom :
            plt.ylim(top=self.test_loss[len(self.test_loss)//factor])
            plt.xlim(left=len(self.test_loss)//factor)

        plt.show()

    # ------------------ Save the network ------------------
    def save(self):
        pass

    # ------------------ Load the network ------------------
    def load(self):
        pass