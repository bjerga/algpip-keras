from keras.models import Sequential
from keras.layers import Dense
from keras.activations import relu, softmax
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

model = Sequential([
    Dense(units=4, activation=relu, input_dim=3),
    Dense(units=2, activation=softmax)
])

model.compile(
    loss=categorical_crossentropy,
    optimizer=Adam()
)
