# Note: you are free to organize your code in the way you find most convenient.
# However, make sure that when your main notebook is run, it executes the steps indicated in the assignment.
import numpy as np
import Functions
import random

'''class Perceptron:
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
'''
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
    '''def calculate(self, input : np.ndarray):
        self.z = self.w.dot(input) + self.b
        self.a = self.activation_function.f(self.z)

        return self.a
    '''

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
        
        self.layers.append(Layer(input_dimension, dimensions[0], Functions.Sigmoid))
    
        for ind, sz in enumerate(dimensions[1:-1]):
            self.layers.append(Layer(dimensions[ind], sz, Functions.Sigmoid))

        self.layers.append(Layer(dimensions[-2], dimensions[-1], Functions.SoftMax))
    
    def forward(self, X:np.ndarray):
        A = X.T

        for l in range(self.layers_count):
            Z = self.layers[l].w.dot(A) + self.layers[l].b

            A = self.layers[l].activation_function.f(Z)

            self.layers[l].a = A
            self.layers[l].z = Z
        return A

    '''def predict(self, input : np.ndarray):
        for cur_layer in self.layers:
            input = cur_layer.calculate(input)

        return input #self.apply_soft_max(input)
    '''
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

        self.layers[self.layers_count - 1].gradient = dW
        self.layers[self.layers_count - 1].b_gradient = db

        # calc all other layers

        for l in range(self.layers_count - 2, 0, -1):
            dZ = dAPrev * self.layers[l].activation_function.d(self.layers[l].z)
            dW = dZ.dot(self.layers[l - 1].a.T) / n
            db = np.sum(dZ, axis=1, keepdims=True) / n

            if l > 0:
                dAPrev = self.layers[l].w.T.dot(dZ)
            self.layers[l].gradient = dW
            self.layers[l].b_gradient = db
        
        dZ = dAPrev * self.layers[0].activation_function.d(self.layers[0].z)

        dW = dZ.dot(input) / n
        db = np.sum(dZ, axis=1, keepdims=True) / n

        self.layers[0].gradient = dW
        self.layers[0].b_gradient = db


    def flush_derivatives(self):
        for layer in self.layers:
            layer.flush_derivative(self.learning_rate)

#    def fit(self, input:np.ndarray, output:np.ndarray, w=None):        
#        one_hot_output = prepare_output(output, 7)
#        for ind, x in enumerate(input):
#            #print(x)
#            y_hat = self.predict(x)
#            self.back_propagate(x, one_hot_output[ind], y_hat)
#            self.flush_derivatives(1)
            
#    def fit(self, input:np.ndarray, output: np.ndarray):
#        history = []
#        shuffled_input = list(enumerate(input))
#        one_hot_output = prepare_output(output)
#        for e in range(epochs):
#            loss = 0
#            tp = 0
#            for ind, x in enumerate(shuffled_input, 1):
#                class_res = self.predict_class(x[1])
#                if class_res == output[x[0]]:
#                    tp += 1
#                for i in range(last_layer_size):
#                    loss += (one_hot_output[x[0]][i] - self.layers[self.layers_count - 1].perceptrons[i].a) ** 2
#            history.append(HistoryDict())
#            random.shuffle(shuffled_input)
#            for ind, x in enumerate(shuffled_input, 1):
#                self.predict(x[1])
#                self.back_propagate(x[1], output[x[0]])
#                if ind % batch_size == 0:
#                    self.flush_derivatives(len(input))
   

    def predict(self, input:np.ndarray):
        A = self.forward(input)
        y_hat = np.argmax(A, axis=0) + 1
        return y_hat

    def predict_acc(self, input:np.ndarray, Y : np.ndarray):
        A = self.forward(input)
        y_hat = np.argmax(A, axis=0) + 1
        accuracy = (y_hat == Y).mean()
        return accuracy

    def fit_2(self, input : np.ndarray, output : np.ndarray, num_classes, epochs):
        one_hot_output = prepare_output(output, num_classes)
        history = []
        print(one_hot_output)
            
        for e in range(epochs):
            self.forward(input)
            self.back_propagate(input, one_hot_output)
            self.flush_derivatives()

        return history

    def fit(self, input:np.ndarray, output: np.ndarray, val_in, val_out, batch_size, num_classes, epochs):
        history = []
        shuffled_input = list(enumerate(input))
        one_hot_output = prepare_output(output, num_classes)

        val_in = list(enumerate(val_in))
        one_hot_val_output = prepare_output(val_out, num_classes)
        for e in range(epochs):
            loss = 0
            tp = 0
            for ind, x in enumerate(shuffled_input, 1):
                class_res = self.predict_class(x[1])
                if class_res == output[x[0]]:
                    tp += 1
                for i in range(num_classes):
                    loss += (one_hot_output[x[0]][i] - self.layers[self.layers_count - 1].a[i]) ** 2
            loss /= len(shuffled_input)
            tp /= len(shuffled_input)

            val_loss = 0
            val_tp = 0
            for ind, x in enumerate(val_in, 1):
                class_res = self.predict_class(x[1])
                if class_res == val_out[x[0]]:
                    val_tp += 1
                for i in range(num_classes):
                    val_loss += (one_hot_val_output[x[0]][i] - self.layers[self.layers_count - 1].a[i]) ** 2
            val_loss /= len(val_in)
            val_tp /= len(val_in)
            history.append(HistoryDict(loss, tp, val_loss, val_tp))

            
            random.shuffle(shuffled_input)
            for ind, x in enumerate(shuffled_input, 1):
                self.predict(x[1])
                self.back_propagate(x[1], output, one_hot_output[x[0]])
                if ind % batch_size == 0:
                    self.flush_derivatives(len(input))
            print(history[-1].loss)
        return history



def prepare_output(output: np.ndarray, num_classes: int):
    result = np.zeros(output.shape[0] * num_classes).reshape(output.shape[0], num_classes)
    for ind, target in enumerate(output):
        result[ind, int(target) - 1] = 1
    return result

'''class HistoryDict(TypedDict):
    loss: List[float]
    acc: List[float]
    val_loss: List[float]
    val_acc: List[float]

def plot_history(history: HistoryDict, title: str, font_size: Optional[int] = 14) -> None:
    plt.suptitle(title, fontsize=font_size)
    ax1 = plt.subplot(121)
    ax1.set_title("Loss")
    ax1.plot(history["loss"], label="train")
    ax1.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch")
    ax1.legend()

    ax2 = plt.subplot(122)
    ax2.set_title("Accuracy")
    ax2.plot(history["acc"], label="train")
    ax2.plot(history["val_acc"], label="val")
    plt.xlabel("Epoch")
    ax2.legend()
    '''
class HistoryDict():
    def __init__(self, loss, acc, val_loss, val_acc):
        self.loss = loss
        self.acc = acc
        self.val_loss = val_loss
        self.val_acc = val_acc

