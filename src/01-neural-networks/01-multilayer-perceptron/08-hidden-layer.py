import numpy as np

np.random.seed(99)

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

hidden_weight = np.random.rand(4, 2) / 2
hidden_bias = np.random.rand(4)

output_weight = np.random.rand(1, 4) / 4
output_bias = np.random.rand(1)


def forward(x, w, b):
    return x @ w.T + b


def mse_loss(p, y):
    return np.mean(np.square(y - p))


def gradient(p, y):
    return -2 * (y - p) / len(y)


def gradient_backward(d, w):
    return d @ w


def backward(x, d, w, b, lr):
    w -= d.T @ x * lr
    b -= np.sum(d, axis=0) * lr
    return w, b


LEARNING_RATE = 0.00001

BATCH_SIZE = 2

EPOCHS = 1000

for epoch in range(EPOCHS):
    for i in range(0, len(train_features), BATCH_SIZE):
        features = train_features[i: i + BATCH_SIZE]
        labels = train_labels[i: i + BATCH_SIZE]

        hidden = forward(features, hidden_weight, hidden_bias)
        predictions = forward(hidden, output_weight, output_bias)

        output_delta = gradient(predictions, labels)
        hidden_delta = gradient_backward(output_delta, output_weight)

        output_weight, output_bias = backward(hidden, output_delta, output_weight, output_bias, LEARNING_RATE)
        hidden_weight, hidden_bias = backward(features, hidden_delta, hidden_weight, hidden_bias, LEARNING_RATE)

print(f'hidden weight: {hidden_weight}')
print(f'hidden bias:   {hidden_bias}')
print(f'output weight: {output_weight}')
print(f'output bias:   {output_bias}')

hidden = forward(test_features, hidden_weight, hidden_bias)
predictions = forward(hidden, output_weight, output_bias)
print(f'predictions: {predictions}')

loss = mse_loss(predictions, test_labels)
print(f'loss: {loss}')
