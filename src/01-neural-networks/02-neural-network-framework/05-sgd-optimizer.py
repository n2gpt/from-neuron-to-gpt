from abc import ABC, abstractmethod

import numpy as np


class Tensor:

    def __init__(self, data):
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data)
        self.gradient_fn = None
        self.parents = set()

    def backward(self):
        if self.gradient_fn is not None:
            self.gradient_fn()

        for parent in self.parents:
            parent.backward()

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f'Tensor({self.data})'


class Layer(ABC):

    def __call__(self, x: Tensor):
        return self.forward(x)

    @abstractmethod
    def forward(self, x: Tensor):
        pass

    @property
    def parameters(self):
        return []

    def __repr__(self):
        return f"{type(self).__name__}[]"


class Loss(ABC):

    def __call__(self, p: Tensor, y: Tensor):
        return self.loss(p, y)

    @abstractmethod
    def loss(self, p: Tensor, y: Tensor):
        pass


class Optimizer(ABC):

    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    @abstractmethod
    def step(self):
        pass


feature = Tensor([28.1, 58.0])
label = Tensor([165])


class Linear(Layer):

    def __init__(self, in_size, out_size):
        self.weight = Tensor(np.ones((out_size, in_size)) / in_size)
        self.bias = Tensor(np.zeros(out_size))

    def forward(self, x: Tensor):
        p = Tensor(x.data @ self.weight.data.T + self.bias.data)

        def gradient_fn():
            self.weight.grad += p.grad * x.data
            self.bias.grad += np.sum(p.grad)

        p.gradient_fn = gradient_fn
        return p

    @property
    def parameters(self):
        return [self.weight, self.bias]

    def __repr__(self):
        return f'Linear[weight{self.weight.shape}; bias{self.bias.shape}]'


class MSELoss(Loss):

    def loss(self, p: Tensor, y: Tensor):
        mse = Tensor(np.mean(np.square(y.data - p.data)))

        def gradient_fn():
            p.grad += -2 * (y.data - p.data)

        mse.gradient_fn = gradient_fn
        mse.parents = {p}
        return mse


class SGDOptimizer(Optimizer):

    def step(self):
        for param in self.parameters:
            param.data -= param.grad * self.lr


LEARNING_RATE = 0.00001

layer = Linear(2, 1)
loss_fn = MSELoss()
optimizer = SGDOptimizer(layer.parameters, lr=LEARNING_RATE)
print(layer)

prediction = layer(feature)
print(f'prediction: {prediction}')

loss = loss_fn(prediction, label)
print(f'loss: {loss}')

loss.backward()
print(f'weight.grad: {layer.weight.grad}')
print(f'bias.grad:   {layer.bias.grad}')

optimizer.step()
print(f'weight: {layer.weight}')
print(f'bias:   {layer.bias}')

prediction = layer(feature)
print(f'prediction: {prediction}')

loss = loss_fn(prediction, label)
print(f'loss: {loss}')
