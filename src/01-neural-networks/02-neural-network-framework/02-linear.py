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


feature = Tensor([28.1, 58.0])


class Linear(Layer):

    def __init__(self, in_size, out_size):
        self.weight = Tensor(np.ones((out_size, in_size)) / in_size)
        self.bias = Tensor(np.zeros(out_size))

    def forward(self, x: Tensor):
        return Tensor(x.data @ self.weight.data.T + self.bias.data)

    def __repr__(self):
        return f'Linear[weight{self.weight.shape}; bias{self.bias.shape}]'


layer = Linear(2, 1)
print(layer)

prediction = layer(feature)
print(f'prediction: {prediction}')
