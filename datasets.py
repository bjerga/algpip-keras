import keras.datasets as _kds
from keras.utils.np_utils import to_categorical


class _Dataset:
    def __init__(self, name, input_shape, load_fn, use_1d=False):
        if use_1d:
            name += '_1d'

            input_shape_1d = 1
            for dim in input_shape:
                input_shape_1d *= dim
            input_shape = input_shape_1d,

        self.name = name
        self.input_shape = input_shape
        self._load_fn = load_fn

    def load_train_data(self):
        (train_data, train_targets), _ = self._load_fn()
        return self._process(train_data, train_targets)

    def load_test_data(self):
        _, (test_data, test_targets) = self._load_fn()
        return self._process(test_data, test_targets)

    def _process(self, data, targets):
        # reshape data (to fit input layer)
        data = data.reshape((data.shape[0],) + self.input_shape)

        # normalize data to [0.0, 1.0]
        data = data.astype('float32')
        data /= 255.0

        # make targets categorical (to fit output layer)
        targets = to_categorical(targets, 10)

        return data, targets


class _CIFAR10(_Dataset):
    def __init__(self, use_1d=False):
        name = 'cifar10'
        input_shape = (32, 32, 3)
        load_fn = _kds.cifar10.load_data
        super(_CIFAR10, self).__init__(name, input_shape, load_fn, use_1d)


class _CIFAR100(_Dataset):
    def __init__(self, use_1d=False):
        name = 'cifar100'
        input_shape = (32, 32, 3)
        load_fn = _kds.cifar100.load_data
        super(_CIFAR100, self).__init__(name, input_shape, load_fn, use_1d)


class _FASHION_MNIST(_Dataset):
    def __init__(self, use_1d=False):
        name = 'fashion_mnist'
        input_shape = (28, 28, 1)
        load_fn = _kds.fashion_mnist.load_data
        super(_FASHION_MNIST, self).__init__(name, input_shape, load_fn, use_1d)


class _MNIST(_Dataset):
    def __init__(self, use_1d=False):
        name = 'mnist'
        input_shape = (28, 28, 1)
        load_fn = _kds.mnist.load_data
        super(_MNIST, self).__init__(name, input_shape, load_fn, use_1d)


cifar10 = _CIFAR10()
cifar10_1d = _CIFAR10(use_1d=True)

cifar100 = _CIFAR100()
cifar100_1d = _CIFAR100(use_1d=True)

fashion_mnist = _FASHION_MNIST()
fashion_mnist_1d = _FASHION_MNIST(use_1d=True)

mnist = _MNIST()
mnist_1d = _MNIST(use_1d=True)
