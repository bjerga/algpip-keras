from keras.datasets import cifar10, cifar100, fashion_mnist, mnist
from keras.utils.np_utils import to_categorical


TRAIN = 'train'
TEST = 'test'

DEFAULT_CIFAR_1D_SHAPE = 3072,
DEFAULT_CIFAR_3D_SHAPE = (32, 32, 3)  # TODO: check tf channel pos in mimir
DEFAULT_MNIST_1D_SHAPE = 784,
DEFAULT_MNIST_2D_SHAPE = (28, 28)

# TODO: må sjekke input shape for de andre datasettene
def get_input_shape():
    return 784,


def cifar10_train_data():
    return load_data(cifar10.load_data, TRAIN)


def cifar10_test_data():
    return load_data(cifar10.load_data, TEST)


def cifar100_train_data():
    return load_data(cifar100.load_data, TRAIN)


def cifar100_test_data():
    return load_data(cifar100.load_data, TEST)


def fashion_mnist_train_data():
    return load_data(fashion_mnist.load_data, TRAIN)


def fashion_mnist_test_data():
    return load_data(fashion_mnist.load_data, TEST)


def mnist_train_data():
    return load_data(mnist.load_data, TRAIN)


def mnist_test_data():
    return load_data(mnist.load_data, TEST)

# TODO: remove default input_shape
def load_data(loader_fn, dataset, input_shape=(784,)):
    (train_data, train_targets), (test_data, test_targets) = loader_fn()

    if dataset == TRAIN:
        data = train_data
        targets = train_targets
    elif dataset == TEST:
        data = test_data
        targets = test_targets
    else:
        raise FileNotFoundError('Chosen dataset could not be found')

    return process(data, targets, input_shape)


def process(data, targets, input_shape):
    # reshape data (to fit input layer)
    data = data.reshape((data.shape[0],) + input_shape)

    # normalize data to [0.0, 1.0]
    data = data.astype('float32')
    data /= 255.0

    # make targets categorical (to fit output layer)
    targets = to_categorical(targets, 10)

    return data, targets
