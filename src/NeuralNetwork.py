# Note: you are free to organize your code in the way you find most convenient.
# However, make sure that when your main notebook is run, it executes the steps indicated in the assignment.
import numpy as np
import Functions

class Perceptron:
    w : np.array
    x : np.array
    b : float
    def __init__(self, initial_weight: np.array, initial_bias: float, activation_function: Functions.ActivationFunction):
        self.w = initial_weight
        self.b = initial_bias
        self.activation_function = activation_function
    
    def forward(self, input):
        self.x = input
        return self.activation_function.f(self.w.dot(input) + self.b)

    def compute_loss(self, pred: float, actual: float):
        return actual - pred

    def update(self, loss, alpha):
        self.w += alpha * loss * self.x
        self.b += alpha * loss
        return 

class Layer:
    def __init__(self, input_size: int, layer_dim: int):
        self.input_size = input_size
        self.layer_dim = layer_dim

class ANN:
    def __init__(self):
        pass

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