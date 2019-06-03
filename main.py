from loader import get_input_shape, mnist_train_data, mnist_test_data

import numpy as np
import tensorflow as tf
from os import mkdir
from os.path import join
from time import time

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.activations import relu, softmax
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy

# removes pestering tensorflow warnings
tf.logging.set_verbosity(tf.logging.ERROR)

MODELS_FOLDER = 'models'
MODEL_EXT = '.h5'


def main():
    start_time = time()

    # create a model, then train and test it.
    model = create()
    model = train(model)
    test(model)
    # test_with_evaluate(model)

    print('\nFinished in {:.2f} seconds'.format(time() - start_time))


def create():
    input_shape = get_input_shape()

    model = Sequential([
        Dense(units=1, activation=relu, input_shape=input_shape),
        Dense(units=10, activation=softmax)
    ])

    # https://keras.io/models/sequential/#compile
    model.compile(
        loss=categorical_crossentropy,
        optimizer=Adam(),
        metrics=[categorical_accuracy]
    )

    return model


def train(model, epochs=10):
    print('\nCommence model training\n')

    train_images, train_targets = mnist_train_data()

    # https://keras.io/models/sequential/#fit
    model.fit(
        x=train_images,
        y=train_targets,
        batch_size=32,
        epochs=epochs,
        verbose=2,
        validation_split=0.1,
        shuffle=True
    )

    print('\nCompleted model training\n')

    return model


def test(model, verbose=False):
    print('\nCommence model testing\n')

    test_images, test_targets = mnist_test_data()

    # test for all indices and count correctly classified
    correctly_classified = 0
    for i in range(len(test_images)):
        # get model classification
        test_image = test_images[i]

        # https://keras.io/models/sequential/#predict
        classification = model.predict(test_image.reshape((1,) + test_image.shape))[0]

        # find correct classification
        correct = test_targets[i]

        # count correctly classified, and print if incorrect
        if np.argmax(classification) == np.argmax(correct):
            correctly_classified += 1
        elif verbose:
            with np.printoptions(precision=3, suppress=True):
                print('Incorrectly classified {} as {}. Classification output: {}'.format(np.argmax(correct),
                                                                                          np.argmax(classification),
                                                                                          classification))

    print('\nModel correctly classified {}/{}.'.format(correctly_classified, len(test_images)))
    print('\nCompleted model testing')


def test_with_evaluate(model, verbose=True):
    print('\nCommence model testing\n')

    test_images, test_targets = mnist_test_data()

    # https://keras.io/models/sequential/#evaluate
    loss_and_metrics = model.evaluate(
        x=test_images,
        y=test_targets,
        verbose=verbose
    )

    print('\nLoss and metrics:', loss_and_metrics)
    print('\nCompleted model testing')


def save(model, filename):
    # make path, if not exists
    try:
        mkdir(MODELS_FOLDER)
    except FileExistsError:
        # file exists, which is what we want
        pass

    path = join(MODELS_FOLDER, filename + MODEL_EXT)
    model.save(path)


def load(filename):
    path = join(MODELS_FOLDER, filename + MODEL_EXT)
    return load_model(path)


main()
