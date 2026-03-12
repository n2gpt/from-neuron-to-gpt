import numpy as np

feature = np.array([28.1, 58.0])
label = np.array([165])

weight = np.ones((1, 2)) / 2
bias = np.zeros(1)


def forward(x, w, b):
    return x @ w.T + b


def mse_loss(p, y):
    return np.mean(np.square(y - p))


prediction = forward(feature, weight, bias)
print(f'prediction: {prediction}')

loss = mse_loss(prediction, label)
print(f'loss: {loss}')
