from abc import ABC, abstractmethod

import numpy as np


class Tensor:

    def __init__(self, data):
        self.data = np.array(data)

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

    def __repr__(self):
        return f"{type(self).__name__}[]"


class Loss(ABC):

    def __call__(self, p: Tensor, y: Tensor):
        return self.loss(p, y)

    @abstractmethod
    def loss(self, p: Tensor, y: Tensor):
        pass


feature = Tensor([28.1, 58.0])
label = Tensor([165])


class Linear(Layer):

    def __init__(self, in_size, out_size):
        self.weight = Tensor(np.ones((out_size, in_size)) / in_size)
        self.bias = Tensor(np.zeros(out_size))

    def forward(self, x: Tensor):
        return Tensor(x.data @ self.weight.data.T + self.bias.data)

    def __repr__(self):
        return f'Linear[weight{self.weight.shape}; bias{self.bias.shape}]'


class MSELoss(Loss):

    def loss(self, p: Tensor, y: Tensor):
        return Tensor(np.mean(np.square(y.data - p.data)))


layer = Linear(2, 1)
loss_fn = MSELoss()
print(layer)

prediction = layer(feature)
print(f'prediction: {prediction}')

loss = loss_fn(prediction, label)
print(f'loss: {loss}')
