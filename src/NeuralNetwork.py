# Note: you are free to organize your code in the way you find most convenient.
# However, make sure that when your main notebook is run, it executes the steps indicated in the assignment.
import numpy as np
import Functions
import random

class Perceptron:
    w : np.ndarray
    x : np.ndarray
    a : float
    b : float
    z : float
    gradient : float
    b_gradient : float
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
    derivatives : np.ndarray
    b_derivatives : np.ndarray
    perceptrons : list[Perceptron]

    def __init__(self, input_size: int, layer_dim: int):
        self.input_size = input_size
        self.layer_dim = layer_dim

        self.derivatives = np.zeros(layer_dim)
        self.b_derivatives = np.zeros(layer_dim
                                      )
        self.perceptrons = [Perceptron(np.random.rand(input_size), 0.1, Functions.Sigmoid) for i in range(layer_dim)]
    def calculate(self, input : np.ndarray):
        output = np.zeros(self.layer_dim)

        for ind, perceptron in enumerate(self.perceptrons):
            output[ind] = perceptron.forward(input)
        return output

    def flush_derivative(self, count:int, learning_rate:float):
        for perceptron in self.perceptrons:
            perceptron.w = perceptron.w + (learning_rate * perceptron.gradient / count)
            perceptron.b = perceptron.b + (learning_rate * perceptron.b_gradient / count)

            perceptron.gradient = 0
            perceptron.b_gradient = 0

class ANN:
    layers : list[Layer]
    layers_count : int
    learning_rate : float
    def __init__(self, input_dimension : int, dimensions : list[int], learning_rate : float):
        self.layers = []
        self.layers_count = len(dimensions)
        self.learning_rate = learning_rate
        
        self.layers.append(Layer(input_dimension, dimensions[0]))
    
        for ind, sz in enumerate(dimensions[1:]):
            self.layers.append(Layer(dimensions[ind], sz))
            

    def calc_derivatives(self, expected_output : np.ndarray):
        # precalculate derivates of last layer
        last_layer_size = len(self.layers[self.layers_count - 1].perceptrons)
        
        for i in range(last_layer_size):
            self.layers[self.layers_count - 1].derivatives[i] = 2 * (expected_output[i] - self.layers[self.layers_count - 1].perceptrons[i].a)
            self.layers[self.layers_count - 1].b_derivatives[i] = 2 * (expected_output[i] - self.layers[self.layers_count - 1].perceptrons[i].a)

        # we start from layer_size - 1, ignoring the output layer
        for l in range(len(self.layers) - 2, -1, -1):
            for j in range(len(self.layers[l].perceptrons)):
                self.layers[l].b_derivatives[j] = 0
                self.layers[l].derivatives[j] = 0
                for i in range(len(self.layers[l + 1].perceptrons)):
                    z_i = self.layers[l + 1].perceptrons[i].z
                    
                    current_addition = self.layers[l + 1].perceptrons[i].w[j]
                    current_addition *= self.layers[l + 1].perceptrons[i].activation_function.d(z_i) 
                    current_addition *= self.layers[l + 1].derivatives[i]

                    self.layers[l].derivatives[j] += current_addition

                    current_addition = self.layers[l + 1].perceptrons[i].b
                    current_addition *= self.layers[l + 1].perceptrons[i].activation_function.d(z_i) 
                    current_addition *= self.layers[l + 1].b_derivatives[i]

                    self.layers[l].b_derivatives[j] += current_addition

    def predict(self, input : np.ndarray):
        for cur_layer in self.layers:
            input = cur_layer.calculate(input)
        return input

    def predict_class(self, output: np.ndarray):
        return np.where(output == np.max(output))[0][0] + 1


    def back_propagate(self, input : np.ndarray, expected_output : np.ndarray):
        self.calc_derivatives(expected_output)
        
        # backpropagate all layers except the first one - as it requires the input
        for l in range(len(self.layers) - 1, 0, -1):
            for j in range(len(self.layers[l].perceptrons)):
                for k in range(len(self.layers[l - 1].perceptrons)):
                    to_calc = self.layers[l].perceptrons[j].activation_function.d(self.layers[l].perceptrons[j].z)
                    to_calc *= self.layers[l].derivatives[j]

                    to_calc *= self.layers[l - 1].perceptrons[k].a
                    
                    self.layers[l].perceptrons[j].gradient += to_calc

                    to_calc = self.layers[l].perceptrons[j].activation_function.d(self.layers[l].perceptrons[j].z)
                    to_calc *= self.layers[l].b_derivatives[j]

                    self.layers[l].perceptrons[j].b_gradient += to_calc
        
        # back propagate layer 0
        for ind2, perceptron in enumerate(self.layers[0].perceptrons):
            for ind in range(len(perceptron.w)):
                to_calc = perceptron.activation_function.d(perceptron.z)
                to_calc *= self.layers[0].derivatives[ind2]
                to_calc *= input[ind]

                perceptron.gradient += to_calc

                to_calc = perceptron.activation_function.d(perceptron.z)
                to_calc *= self.layers[0].derivatives[ind2]

                perceptron.b_gradient += to_calc

    def flush_derivatives(self, count:int):
        for layer in self.layers:
            layer.flush_derivative(count, self.learning_rate)

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
                    loss += (one_hot_output[x[0]][i] - self.layers[self.layers_count - 1].perceptrons[i].a) ** 2
            loss /= len(shuffled_input)
            tp /= len(shuffled_input)

            val_loss = 0
            val_tp = 0
            for ind, x in enumerate(val_in, 1):
                class_res = self.predict_class(x[1])
                if class_res == val_out[x[0]]:
                    val_tp += 1
                for i in range(num_classes):
                    val_loss += (one_hot_val_output[x[0]][i] - self.layers[self.layers_count - 1].perceptrons[i].a) ** 2
            val_loss /= len(val_in)
            val_tp /= len(val_in)
            history.append(HistoryDict(loss, tp, val_loss, val_tp))

            
            random.shuffle(shuffled_input)
            for ind, x in enumerate(shuffled_input, 1):
                self.predict(x[1])
                self.back_propagate(x[1], one_hot_output[x[0]])
                if ind % batch_size == 0:
                    self.flush_derivatives(len(input))
            print(history[-1].loss)
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

'''
class GCN(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_classes: int,
        hidden_dim: int = 16
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x: Tensor, edge_index: Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
'''''''''''''''
