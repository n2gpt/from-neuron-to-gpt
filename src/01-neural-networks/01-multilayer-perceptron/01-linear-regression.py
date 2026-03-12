import numpy as np

feature = np.array([28.1, 58.0])
print(feature)

weight = np.ones((1, 2)) / 2
bias = np.zeros(1)
print(f'weight: {weight}')
print(f'bias:   {bias}')


def forward(x, w, b):
    return x @ w.T + b


prediction = forward(feature, weight, bias)
print(f'prediction: {prediction}')
