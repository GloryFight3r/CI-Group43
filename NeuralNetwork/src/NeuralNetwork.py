# Note: you are free to organize your code in the way you find most convenient.
# However, make sure that when your main notebook is run, it executes the steps indicated in the assignment.
import numpy as np
import Functions
import random

class Perceptron:
    def __init__(self, initial_weight: np.ndarray, initial_bias: float, activation_function: Functions.ActivationFunction):
        self.w = initial_weight
        self.b = initial_bias
        self.a = 0
        self.gradient = 0
        self.b_gradient = 0
        self.activation_function = activation_function
    
    def forward(self, input : np.ndarray):
        self.x = input
        self.z = self.w.dot(input) + self.b
        self.a = self.activation_function.f(self.z)
        return self.a

    def compute_loss(self, pred: float, actual: float):
        return actual - pred

    def update(self, loss, alpha):
        self.w += alpha * loss * self.x
        self.b += alpha * loss
        return 

class Layer:
    input_size : int
    layer_dim : int

    w : np.ndarray
    b : np.ndarray

    a : np.ndarray
    z : np.ndarray

    gradient : np.ndarray
    b_gradient : np.ndarray

    activation_function : Functions.ActivationFunction

    def __init__(self, input_size: int, layer_dim: int, activation_function: Functions.ActivationFunction):
        self.input_size = input_size
        self.layer_dim = layer_dim

        self.w = np.random.randn(layer_dim, input_size) / np.sqrt(input_size)
        self.b = np.zeros((layer_dim, 1))
        self.activation_function = activation_function
        
        self.gradient = np.zeros(layer_dim * input_size).reshape(layer_dim, input_size)
 
        self.b_gradient = np.zeros(layer_dim).reshape(-1, 1)

    def flush_derivative(self, learning_rate:float):
        self.w = self.w - (learning_rate * self.gradient)
        self.b = self.b - (learning_rate * self.b_gradient)

        self.gradient = np.zeros(self.layer_dim * self.input_size).reshape(self.layer_dim, self.input_size)
        self.b_gradient = np.zeros(self.layer_dim).reshape(-1, 1)

class ANN:
    layers : list[Layer]
    layers_count : int
    learning_rate : float
    def __init__(self, input_dimension : int, dimensions : list[int], learning_rate : float):
        self.layers = []
        self.layers_count = len(dimensions)
        self.learning_rate = learning_rate
        
        prev = input_dimension
        for sz in dimensions[0:-1]:
            self.layers.append(Layer(prev, sz, Functions.Sigmoid))
            prev = sz
        
        # last layer user softmax activation function
        self.layers.append(Layer(prev, dimensions[-1], Functions.SoftMax))
    
    def forward(self, X:np.ndarray):
        """
        Forward propagates the matrix containing data records for the current input X
        """
        A = X.T

        for l in range(self.layers_count):
            Z = self.layers[l].w.dot(A) + self.layers[l].b

            A = self.layers[l].activation_function.f(Z)

            self.layers[l].a = A
            self.layers[l].z = Z
        return A

    def apply_soft_max(self, output):
        return np.exp(output) / np.sum(np.exp(output))

    def predict_class(self, output: np.ndarray):
        return np.where(output == np.max(output))[0][0] + 1

    def back_propagate(self, input : np.ndarray, expected_output : np.ndarray):
        """
        Parameters
        -------------
        input : matrix containing input
        expected_output : matrix containing output in a one-hot encoded format
        """
        n = input.shape[0]

        # calculate derivative of the cross entropy loss function on the last layer
        dZ = self.layers[-1].a - expected_output.T

        # last layer is softmax
        dW = dZ.dot(self.layers[-2].a.T) / n
        db = np.sum(dZ, axis=1, keepdims=True) / n
        dAPrev = self.layers[-1].w.T.dot(dZ)

        self.layers[self.layers_count - 1].gradient += dW
        self.layers[self.layers_count - 1].b_gradient += db

        # calc all other layers except the last and first layers
        for l in range(self.layers_count - 2, 0, -1):
            dZ = dAPrev * self.layers[l].activation_function.d(self.layers[l].z)
            dW = dZ.dot(self.layers[l - 1].a.T) / n
            db = np.sum(dZ, axis=1, keepdims=True) / n

            if l > 0:
                dAPrev = self.layers[l].w.T.dot(dZ)
            self.layers[l].gradient += dW
            self.layers[l].b_gradient += db
        
        # calculate derivative for the first layer - we don't techincally have an input layer so we have to do this
        dZ = dAPrev * self.layers[0].activation_function.d(self.layers[0].z)

        dW = dZ.dot(input) / n
        db = np.sum(dZ, axis=1, keepdims=True) / n

        self.layers[0].gradient += dW
        self.layers[0].b_gradient += db


    def flush_derivatives(self):
        for layer in self.layers:
            layer.flush_derivative(self.learning_rate)
   

    def predict(self, input:np.ndarray):
        A = self.forward(input)
        y_hat = np.argmax(A, axis=0) + 1
        return y_hat

    def predict_acc(self, input:np.ndarray, Y : np.ndarray):
        A = self.forward(input)
        y_hat = np.argmax(A, axis=0) + 1
        accuracy = (y_hat == Y).mean()
        return accuracy

    def confusion_matrix(self, input:np.ndarray, Y : np.ndarray, num_classes):
        matrix = [ [0 for j in range(num_classes)] for i in range(num_classes) ]
        A = self.forward(input)
        y_hat = np.argmax(A, axis=0) + 1
        for i in range(len(y_hat)):
            matrix[int(y_hat[i] - 1)][int(Y[i] - 1)] += 1
        return matrix

    def fit(self, input : np.ndarray, output : np.ndarray, val_in, val_output, num_classes, epochs, early_stop):
        """
        Parameters 
        -----------
        input - matrix containing data records per row with features as columns of training data set
        output - array containing labels for data records of training data set 
        val_in - matrix containing data records per row with features as columns of validation data set
        val_output - array containing labels for data records of validation data set
        """

        one_hot_output = prepare_output(output, num_classes)
        one_hot_val_output = prepare_output(val_output, num_classes)
        
        history = []
        previous_acc = 0 # previous accuracy we had achieved early_stop epochs before
        eps = 0.001 # difference between accuracy for validation data set for which we stop training
        for e in range(epochs):
            self.forward(input) # forward propagation of input
            self.back_propagate(input, one_hot_output) # backward propagation of loss 
            self.flush_derivatives() # flush the gradient for the weights/biases

            val_accuracy, val_loss = self.loss_and_accuracy(val_in, val_output, one_hot_val_output)
            train_accuracy, train_loss = self.loss_and_accuracy(input, output, one_hot_output)
            history.append(HistoryDict(train_loss, train_accuracy, val_loss, val_accuracy))
            
            if e % early_stop == 0:
                val_accuracy, val_loss = self.loss_and_accuracy(val_in, val_output, one_hot_val_output)
                if val_accuracy - previous_acc < eps: # check whether we should stop training
                    break

                previous_acc = val_accuracy

        return history

    def loss_and_accuracy(self, input, output, one_hot_output):
        oh = one_hot_output.T
        A = self.forward(input)

        y_hat = np.argmax(A, axis=0) + 1
        train_accuracy = (y_hat == output).mean()

        #Cross entropy Loss
        train_loss = -np.mean(one_hot_output * np.log(A.T + 1e-8))

        return (train_accuracy, train_loss)

def prepare_output(output: np.ndarray, num_classes: int):
    result = np.zeros(output.shape[0] * num_classes).reshape(output.shape[0], num_classes)
    for ind, target in enumerate(output):
        result[ind, int(target) - 1] = 1
    return result

class HistoryDict():
    def __init__(self, loss, acc, val_loss, val_acc):
        self.loss = loss
        self.acc = acc
        self.val_loss = val_loss
        self.val_acc = val_acc

    def __str__(self):
        return f'({self.loss}, {self.acc}, {self.val_loss}, {self.val_acc})' 

    def __repr__(self):
        return self.__str__()


