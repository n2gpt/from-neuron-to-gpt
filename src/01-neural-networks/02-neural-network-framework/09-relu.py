from abc import ABC, abstractmethod

import numpy as np

np.random.seed(99)


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


class Dataset(ABC):

    def __init__(self, batch_size=1):
        self.batch_size = batch_size

        self.test_features = None
        self.test_labels = None
        self.train_features = None
        self.train_labels = None

        self.load()

        self.features = self.train_features
        self.labels = self.train_labels

    @abstractmethod
    def load(self):
        pass

    def train(self):
        self.features = self.train_features
        self.labels = self.train_labels

    def eval(self):
        self.features = self.test_features
        self.labels = self.test_labels

    def items(self):
        return Tensor(self.features), Tensor(self.labels)

    def __len__(self):
        return len(self.features) // self.batch_size

    def __getitem__(self, index):
        start = index * self.batch_size
        end = start + self.batch_size

        feature = Tensor(self.features[start: end])
        label = Tensor(self.labels[start: end])
        return feature, label


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

    def reset(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)

    @abstractmethod
    def step(self):
        pass


class Model(ABC):

    def __init__(self, layer, loss_fn, optimizer):
        self.layer = layer
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    @abstractmethod
    def train(self, dataset, epochs):
        pass

    @abstractmethod
    def test(self, dataset):
        pass


class IcecreamDataset(Dataset):

    def load(self):
        self.train_features = [[22.5, 72.0],
                               [31.4, 45.0],
                               [19.8, 85.0],
                               [27.6, 63.0]]
        self.train_labels = [[95],
                             [210],
                             [70],
                             [155]]

        self.test_features = [[28.1, 58.0]]
        self.test_labels = [[165]]


class Linear(Layer):

    def __init__(self, in_size, out_size):
        self.weight = Tensor(np.random.rand(out_size, in_size) / in_size)
        self.bias = Tensor(np.random.rand(out_size))

    def forward(self, x: Tensor):
        p = Tensor(x.data @ self.weight.data.T + self.bias.data)

        def gradient_fn():
            self.weight.grad += p.grad.T @ x.data
            self.bias.grad += np.sum(p.grad, axis=0)
            x.grad += p.grad @ self.weight.data

        p.gradient_fn = gradient_fn
        p.parents = {x}
        return p

    @property
    def parameters(self):
        return [self.weight, self.bias]

    def __repr__(self):
        return f'Linear[weight{self.weight.shape}; bias{self.bias.shape}]'


class Sequential(Layer):

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x: Tensor):
        for l in self.layers:
            x = l(x)
        return x

    @property
    def parameters(self):
        return [p for l in self.layers for p in l.parameters]

    def __repr__(self):
        return '\n'.join(str(l) for l in self.layers if str(l))


class ReLU(Layer):

    def forward(self, x: Tensor):
        a = Tensor(np.maximum(0, x.data))

        def gradient_fn():
            x.grad += a.grad * (a.data > 0)

        a.gradient_fn = gradient_fn
        a.parents = {x}
        return a


class MSELoss(Loss):

    def loss(self, p: Tensor, y: Tensor):
        mse = Tensor(np.mean(np.square(y.data - p.data)))

        def gradient_fn():
            p.grad += -2 * (y.data - p.data) / len(y.data)

        mse.gradient_fn = gradient_fn
        mse.parents = {p}
        return mse


class SGDOptimizer(Optimizer):

    def step(self):
        for param in self.parameters:
            param.data -= param.grad * self.lr


class NNModel(Model):

    def train(self, dataset, epochs):
        dataset.train()

        for epoch in range(epochs):
            for i in range(len(dataset)):
                features, labels = dataset[i]

                self.optimizer.reset()
                predictions = self.layer(features)
                loss = self.loss_fn(predictions, labels)
                loss.backward()
                self.optimizer.step()

    def test(self, dataset):
        dataset.eval()

        features, labels = dataset.items()
        predictions = self.layer(features)
        loss = self.loss_fn(predictions, labels)
        return predictions, loss


LEARNING_RATE = 0.00001

BATCH_SIZE = 2

EPOCHS = 1000

dataset = IcecreamDataset(BATCH_SIZE)

layer = Sequential([
    Linear(2, 4),
    ReLU(),
    Linear(4, 1),
])
loss_fn = MSELoss()
optimizer = SGDOptimizer(layer.parameters, lr=LEARNING_RATE)

model = NNModel(layer, loss_fn, optimizer)
print(model.layer)

model.train(dataset, EPOCHS)

predictions, loss = model.test(dataset)
print(f'prediction: {predictions}')
print(f'loss:       {loss}')
