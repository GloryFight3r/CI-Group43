# Note: you are free to organize your code in the way you find most convenient.
# However, make sure that when your main notebook is run, it executes the steps indicated in the assignment.
import numpy as np
import Functions

class Perceptron:
    w : np.ndarray
    x : np.ndarray
    a : float
    b : float
    z : float
    gradient : float
    def __init__(self, initial_weight: np.ndarray, initial_bias: float, activation_function: Functions.ActivationFunction):
        self.w = initial_weight
        self.b = initial_bias
        self.gradient = 0
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
    perceptrons : list[Perceptron]

    def __init__(self, input_size: int, layer_dim: int):
        self.input_size = input_size
        self.layer_dim = layer_dim

        self.derivatives = np.zeros(layer_dim)
        self.perceptrons = [Perceptron(np.zeros(input_size), 0, Sigmoid) for i in range(layer_dim)]
    def calculate(self, input : np.ndarray):
        output = np.zeros(self.layer_dim)

        for ind, perceptron in enumerate(self.perceptrons):
            output[ind] = perceptron.forward(input)
        return output

class ANN:
    layers : list[Layer]
    layers_count : int
    def __init__(self, dimensions):
        self.layers = []
        self.layers_count = len(dimensions)
        for ind, sz in enumerate(dimensions):
            self.layers.append(Layer(dimensions[ind - 1], sz))
            

    def calc_derivatives(self, expected_output : np.ndarray):
        # precalculate derivates of last layer
        last_layer_size = len(self.layers[self.layers_count - 1])
        for i in range(last_layer_size):
            self.layers[self.layers_count - 1].derivatives[i] = 2 * (expected_output[i] - self.layers[self.layers_count - 1].perceptrons[i].a)

        # we start from layer_size - 1, ignoring the output layer
        for l in range(len(self.layers) - 2, 0, -1):
            for j in range(len(self.layers[l].perceptrons)):
                for i in range(len(self.layers[l + 1].perceptrons)):
                    z_i = self.layers[l + 1].perceptrons[i].z
                    
                    current_addition = 0
                    current_addition = self.layers[l + 1].perceptrons[i].w[j]
                    current_addition *= self.layers[l + 1].perceptrons[i].activation_function.d(z_i) 
                    current_addition *= self.layers[l + 1].derivatives[i]
                    
                    self.layers[l].derivatives[j] += current_addition
    def predict(self, input : np.ndarray):
        for cur_layer in self.layers:
            input = cur_layer.calculate(input)
        return input

    def back_propagate(self, input : np.ndarray, expected_output : np.ndarray, learning_rate : float):
        self.calc_derivatives(expected_output)
        
        # backpropagate all layers except the first one - as it requires the input
        for l in range(len(self.layers) - 1, 1, -1):
            for j in range(len(self.layers[l].perceptrons)):
                for k in range(len(self.layers[l - 1].perceptrons)):
                    to_calc = self.layers[l - 1].perceptrons[k].a
                    to_calc *= self.layers[l].perceptrons[j].z
                    to_calc *= self.layers[l].derivatives[j]
                    
                    self.layers[l].perceptrons[j].gradient += to_calc * learning_rate
        
        # back propagate layer 0
        for perceptron in self.layers[0].perceptrons:
            for ind in range(len(perceptron.w)):
                to_calc = input[ind]
                to_calc *= self.layers[0].perceptrons[ind].z
                to_calc *= self.layers[0].derivatives[ind]
                    
                perceptron.gradient += to_calc * learning_rate
            
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
