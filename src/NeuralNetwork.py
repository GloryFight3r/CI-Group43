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
    #x : np.ndarray
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

        self.layers.append(Layer(prev, dimensions[-1], Functions.SoftMax))
    
    def forward(self, X:np.ndarray):
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
        #self.calc_derivatives(expected_output, actual_output)
        n = input.shape[0]

        dZ = self.layers[-1].a - expected_output.T

        # last layer is softmax
        dW = dZ.dot(self.layers[-2].a.T) / n
        db = np.sum(dZ, axis=1, keepdims=True) / n
        dAPrev = self.layers[-1].w.T.dot(dZ)

        self.layers[self.layers_count - 1].gradient += dW
        self.layers[self.layers_count - 1].b_gradient += db

        # calc all other layers

        for l in range(self.layers_count - 2, 0, -1):
            dZ = dAPrev * self.layers[l].activation_function.d(self.layers[l].z)
            dW = dZ.dot(self.layers[l - 1].a.T) / n
            db = np.sum(dZ, axis=1, keepdims=True) / n

            if l > 0:
                dAPrev = self.layers[l].w.T.dot(dZ)
            self.layers[l].gradient += dW
            self.layers[l].b_gradient += db
        
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

    def fit_2_improved(self, input : np.ndarray, output : np.ndarray, val_in, val_output, num_classes, epochs):
        one_hot_output = prepare_output(output, num_classes)
        one_hot_val_output = prepare_output(val_output, num_classes)
        
        history = []
            
        for e in range(epochs):
            self.forward(input)
            self.back_propagate(input, one_hot_output)
            self.flush_derivatives()
            if e % 5 == 0:
                train_accuracy, train_loss = self.loss_and_accuracy(input, output, one_hot_output)
                val_accuracy, val_loss = self.loss_and_accuracy(val_in, val_output, one_hot_val_output)
                history.append(HistoryDict(train_loss, train_accuracy, val_loss, val_accuracy))

        return history

    def loss_and_accuracy(self, input, output, one_hot_output):
        oh = one_hot_output.T
        A = self.forward(input)

        y_hat = np.argmax(A, axis=0) + 1
        train_accuracy = (y_hat == output).mean()

        #L2 Loss
        temp = A - oh
        temp = temp ** 2
        train_loss = 0
        for vector in temp:
            train_loss += np.sum(vector) / A.shape[1]

        return (train_accuracy, train_loss)

    def fit(self, input : np.ndarray, output : np.ndarray, val_input : np.ndarray, 
            val_output : np.ndarray, num_classes : int, epochs : int, early_stop : float):

        one_hot_output = prepare_output(output, num_classes)
        val_one_hot_output = prepare_output(val_output, num_classes)
        history = []
        #print(one_hot_output)
        previous_acc = 0
        for e in range(epochs):

            A = self.forward(input)
            
            y_hat = np.argmax(A, axis=0) + 1
            accuracy = (y_hat == output).mean()

            print(accuracy, previous_acc)
            if e % 100 == 0:
                if accuracy - previous_acc < early_stop:
                    break

                previous_acc = accuracy
            
            loss = -np.mean(one_hot_output * np.log(A.T + 1e-8))
           
            self.back_propagate(input, one_hot_output)

            A2 = self.forward(val_input)

            y_hat = np.argmax(A2, axis=0) + 1
            val_accuracy = (y_hat == val_output).mean()

            val_loss = -np.mean(val_one_hot_output * np.log(A2.T + 1e-8))

            history.append(HistoryDict(loss, accuracy, val_loss, val_accuracy))

            self.flush_derivatives()

        return history


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


