import numpy as np

train_features = np.array([[22.5, 72.0],
                           [31.4, 45.0],
                           [19.8, 85.0],
                           [27.6, 63.0]])
train_labels = np.array([[95],
                         [210],
                         [70],
                         [155]])

test_features = np.array([[28.1, 58.0]])
test_labels = np.array([[165]])

weight = np.ones((1, 2)) / 2
bias = np.zeros(1)


def forward(x, w, b):
    return x @ w.T + b


def mse_loss(p, y):
    return np.mean(np.square(y - p))


def gradient(p, y):
    return -2 * (y - p)


def backward(x, d, w, b, lr):
    w -= d * x * lr
    b -= d * lr
    return w, b


LEARNING_RATE = 0.00001

for i in range(len(train_features)):
    feature = train_features[i]
    label = train_labels[i]

    prediction = forward(feature, weight, bias)
    delta = gradient(prediction, label)
    weight, bias = backward(feature, delta, weight, bias, LEARNING_RATE)

print(f'weight: {weight}')
print(f'bias:   {bias}')

predictions = forward(test_features, weight, bias)
print(f'predictions: {predictions}')

loss = mse_loss(predictions, test_labels)
print(f'loss: {loss}')
