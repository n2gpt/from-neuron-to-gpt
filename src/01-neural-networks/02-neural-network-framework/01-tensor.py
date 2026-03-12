import numpy as np


class Tensor:

    def __init__(self, data):
        self.data = np.array(data)

    def __repr__(self):
        return f'Tensor({self.data})'


feature = Tensor([28.1, 58.0])
print(feature)

weight = Tensor(np.ones((1, 2)) / 2)
bias = Tensor(np.zeros(1))
print(f'weight: {weight}')
print(f'bias:   {bias}')


def forward(x, w, b):
    return Tensor(x.data @ w.data.T + b.data)


prediction = forward(feature, weight, bias)
print(f'prediction: {prediction}')
