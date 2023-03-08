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
    b_gradient : float
    def __init__(self, initial_weight: np.ndarray, initial_bias: float, activation_function: Functions.ActivationFunction):
        self.w = initial_weight
        self.b = initial_bias
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

    def predict_class(self, input: np.ndarray):
        return sorted(list(enumerate(predict(input), 1)), lambda a: a[0])[-1][1]

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

    def fit(self, input:np.ndarray, output: np.ndarray):
        history = []
        shuffled_input = list(enumerate(input))
        one_hot_output = prepare_output(output)
        for e in range(epochs):
            loss = 0
            tp = 0
            for ind, x in enumerate(shuffled_input, 1):
                class_res = self.predict_class(x[1])
                if class_res == output[x[0]]:
                    tp += 1
                for i in range(last_layer_size):
                    loss += (one_hot_output[x[0]][i] - self.layers[self.layers_count - 1].perceptrons[i].a) ** 2
            history.append(HistoryDict())
            random.shuffle(shuffled_input)
            for ind, x in enumerate(shuffled_input, 1):
                self.predict(x[1])
                self.back_propagate(x[1], output[x[0]])
                if ind % batch_size == 0:
                    self.flush_derivatives(len(input))

def prepare_output(output: np.array, num_classes: int):
    result = []
    for target in output:
        result.append(np.zeros(num_classes))
        result[-1][target] = 1
    return np.asarray(result)


class HistoryDict(TypedDict):
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
