import math
import random


# matrix class - basic 2d array stuff
class Matrix:
    def __init__(self, rows, cols, data=None):
        self.rows = rows
        self.cols = cols
        if data is not None:
            self.data = data
        else:
            self.data = [[0.0 for _ in range(cols)] for _ in range(rows)]

    @staticmethod
    def from_list(lst):
        return Matrix(1, len(lst), [lst])

    def to_list(self):
        return self.data[0]

    def copy(self):
        return Matrix(self.rows, self.cols, [row[:] for row in self.data])

    def randomize(self, scale=1.0):
        self.data = [
            [random.uniform(-scale, scale) for _ in range(self.cols)]
            for _ in range(self.rows)
        ]

    # matrix multiplication
    def __matmul__(self, other):
        other_t = list(zip(*other.data))
        return Matrix(self.rows, other.cols, [
            [sum(a * b for a, b in zip(row, col)) for col in other_t]
            for row in self.data
        ])

    # add two matrices (or add scalar)
    def __add__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.rows, self.cols, [
                [a + b for a, b in zip(row1, row2)]
                for row1, row2 in zip(self.data, other.data)
            ])
        return Matrix(self.rows, self.cols, [
            [a + other for a in row]
            for row in self.data
        ])

    def __mul__(self, scalar):
        return Matrix(self.rows, self.cols, [
            [a * scalar for a in row]
            for row in self.data
        ])

    # apply a function to every element
    def apply(self, fn):
        return Matrix(self.rows, self.cols, [
            [fn(a) for a in row]
            for row in self.data
        ])

    def flatten(self):
        return [item for row in self.data for item in row]

    @staticmethod
    def from_flat(data, rows, cols):
        return Matrix(rows, cols, [
            data[i * cols:(i + 1) * cols]
            for i in range(rows)
        ])


# activation functions
def sigmoid(x):
    # Fast algebraic approximation of sigmoid
    return 0.5 * (x / (1.0 + abs(x))) + 0.5

def tanh(x):
    return math.tanh(x)


# simple feedforward net
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            # xavier init-ish
            w = Matrix(layer_sizes[i], layer_sizes[i + 1])
            w.randomize(scale=1.0 / math.sqrt(layer_sizes[i]))
            b = Matrix(1, layer_sizes[i + 1])
            b.randomize(scale=0.1)
            self.weights.append(w)
            self.biases.append(b)

    # forward pass through all layers
    def forward(self, inputs):
        x = inputs
        for i in range(len(self.weights)):
            w_t = list(zip(*self.weights[i].data))
            b = self.biases[i].data[0]
            
            x = [sum(a * w for a, w in zip(x, col)) + bias for col, bias in zip(w_t, b)]
            
            if i < len(self.weights) - 1:
                x = [math.tanh(v) for v in x]
            else:
                x = [sigmoid(v) for v in x]
        return x

    # get all params as 1 big list (for GA)
    def get_params(self):
        params = []
        for w in self.weights:
            params.extend(w.flatten())
        for b in self.biases:
            params.extend(b.flatten())
        return params

    # set all params from 1 big list
    def set_params(self, params):
        idx = 0
        for i in range(len(self.weights)):
            rows = self.weights[i].rows
            cols = self.weights[i].cols
            size = rows * cols
            self.weights[i] = Matrix.from_flat(params[idx:idx + size], rows, cols)
            idx += size
        for i in range(len(self.biases)):
            rows = self.biases[i].rows
            cols = self.biases[i].cols
            size = rows * cols
            self.biases[i] = Matrix.from_flat(params[idx:idx + size], rows, cols)
            idx += size

    # makes a copy with same weights
    def copy(self):
        nn = NeuralNetwork(self.layer_sizes)
        nn.set_params(self.get_params())
        return nn
