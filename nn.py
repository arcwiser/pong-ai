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


# cache math.tanh as a local for speed
_tanh = math.tanh


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
        self._cache_weights()

    def _cache_weights(self):
        # flatten all weights into flat tuples for zero-overhead access
        # weights_flat[layer] = tuple of (col_tuples, bias_tuple)
        self._wf = []
        for i in range(len(self.weights)):
            cols = tuple(tuple(col) for col in zip(*self.weights[i].data))
            bias = tuple(self.biases[i].data[0])
            self._wf.append((cols, bias))
        self._n_layers = len(self.weights)

    # forward pass - absolute maximum speed pure python
    def forward(self, inputs):
        x = inputs
        wf = self._wf
        n = self._n_layers
        last = n - 1
        tanh = _tanh
        for i in range(n):
            cols, bias = wf[i]
            # manual dot products with no generator, no zip, no sum overhead
            if i < last:
                # hidden layer with tanh
                x = [tanh(sum(x[j] * col[j] for j in range(len(x))) + b) for col, b in zip(cols, bias)]
            else:
                # output layer with inline algebraic sigmoid
                out = []
                for col, b in zip(cols, bias):
                    v = sum(x[j] * col[j] for j in range(len(x))) + b
                    out.append(0.5 * (v / (1.0 + abs(v))) + 0.5)
                x = out
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
        self._cache_weights()

    # makes a copy with same weights
    def copy(self):
        nn = NeuralNetwork(self.layer_sizes)
        nn.set_params(self.get_params())
        return nn
