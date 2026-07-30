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
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = random.uniform(-scale, scale)

    # matrix multiplication
    def __matmul__(self, other):
        result = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                s = 0.0
                for k in range(self.cols):
                    s += self.data[i][k] * other.data[k][j]
                result.data[i][j] = s
        return result

    # add two matrices (or add scalar)
    def __add__(self, other):
        if isinstance(other, Matrix):
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] + other.data[i][j]
            return result
        result = self.copy()
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] += other
        return result

    def __mul__(self, scalar):
        result = self.copy()
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] *= scalar
        return result

    # apply a function to every element
    def apply(self, fn):
        result = self.copy()
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = fn(self.data[i][j])
        return result

    def flatten(self):
        flat = []
        for i in range(self.rows):
            for j in range(self.cols):
                flat.append(self.data[i][j])
        return flat

    @staticmethod
    def from_flat(data, rows, cols):
        m = Matrix(rows, cols)
        idx = 0
        for i in range(rows):
            for j in range(cols):
                m.data[i][j] = data[idx]
                idx += 1
        return m


# activation functions
def sigmoid(x):
    if x < -20:
        return 0.0
    if x > 20:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))

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
        x = Matrix.from_list(inputs)
        for i in range(len(self.weights)):
            x = (x @ self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:
                x = x.apply(tanh)  # hidden layers use tanh
            else:
                x = x.apply(sigmoid)  # output uses sigmoid (binary)
        return x.to_list()

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
