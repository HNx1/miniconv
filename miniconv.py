import numpy as np
import math


class GradFloat():
    def __init__(self, val, _children=[]):
        self.val = val
        self.grad = 0
        self._backward = lambda: None
        self._children = set(_children)

    def __repr__(self):
        return str(round(self.val, 2))

    def __add__(self, other):
        other = other if isinstance(other, GradFloat) else GradFloat(other)
        out = GradFloat(self.val+other.val, _children=(self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self+other

    def __mul__(self, other):
        other = other if isinstance(other, GradFloat) else GradFloat(other)
        out = GradFloat(self.val*other.val, _children=(self, other))

        def _backward():
            self.grad += out.grad*other.val
            other.grad += out.grad*self.val
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self*other

    def __sub__(self, other):
        return self.__add__(-other)

    def __pow__(self, other):
        out = GradFloat(self.val**other, _children=(self,))

        def _backward():
            self.grad += other*self.val**(other-1)*out.grad
        out._backward = _backward
        return out

    def tanh(self):
        out = GradFloat((math.exp(2*self.val)-1) /
                        (math.exp(2*self.val)+1), _children=(self,))

        def _backward():
            self.grad = (1-out.val**2)*out.grad
        out._backward = _backward
        return out

    def backward(self):
        self.grad = 1.0
        topList = []
        seen = set()

        def topSort(v):
            if v not in seen:
                seen.add(v)
                for child in v._children:
                    topSort(child)
                topList.append(v)
        topSort(self)
        for item in reversed(topList):
            item._backward()


class ConvolutionNeuron():
    def __init__(self, inputSize, stride=1):
        self.kernel = np.array([GradFloat(np.random.uniform(-1, 1))
                                for _ in range(inputSize**2)]).reshape(inputSize, -1)
        self.b = GradFloat(np.random.uniform(-1, 1))
        self.padShape = (inputSize//2, inputSize//2)
        self.convSize = inputSize

    def __call__(self, x):
        return self.conv(x)

    def parameters(self):
        params = list(self.kernel.flatten())
        params.append(self.b)
        return params

    def pad(self, arr):
        x, y = arr.shape[0], arr.shape[1]
        offx, offy = self.padShape
        res = np.zeros((x+2*offx, y+2*offy), dtype=GradFloat)
        res[offx:x+offx, offy:y+offy] = arr
        return res

    def unitConv(self, arr):
        return np.sum(arr*self.kernel)

    def conv(self, arr):
        s = arr.shape
        arr = self.pad(arr)
        return np.array([(self.unitConv(arr[i:i+self.convSize, j:j+self.convSize]))for i in range(s[0]) for j in range(s[1])], dtype=GradFloat)


class Layer():
    def __init__(self, inputSize, outputSize):
        self.neurons = [ConvolutionNeuron(
            inputSize)for _ in range(outputSize)]

    def __call__(self, x):
        s = np.array(x).shape
        return [np.array([x.tanh() for x in np.sum([n(xi) for xi in x], axis=0, initial=n.b)]).reshape((s[1], s[2]))
                for n in self.neurons]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class CNN():
    def __init__(self, kernelSizeList, featureList, linearInputSize):
        self.layers = [Layer(x, y)
                       for x, y in zip(kernelSizeList, featureList)]
        self.w = np.array([GradFloat(np.random.uniform(-1, 1))
                           for _ in range(linearInputSize**2)]).reshape(linearInputSize, -1)
        self.b = GradFloat(np.random.uniform(-1, 1))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return np.sum(self.w*x) + self.b

    def parameters(self):
        params = [p for layer in self.layers for p in layer.parameters()]
        params.extend(list(self.w.flatten())+[self.b])
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0


def loss(model, data, truth):
    return sum(((y1-y2)**2) for y1, y2 in zip([model(x)for x in data], truth))


def train(model, data, truth, lr):
    model.zero_grad()
    trainLoss = loss(model, data, truth)
    trainLoss.backward()
    for p in model.parameters():
        p.val -= lr * p.grad


def test(model, data, truth):
    testLoss = loss(model, data, truth)
    print(f"Loss was {testLoss.val:.4f}")


def cycle(model, trainData, trainTruth, testData, testTruth, epochs=100, lr=1e-4):
    for _ in range(epochs):
        train(model, trainData, trainTruth, lr)
        test(model, testData, testTruth)
