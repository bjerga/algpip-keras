from keras.models import Model
from keras.layers import Input, Dense
from keras.activations import relu, softmax
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

inputs = Input(shape=(3,))
x = Dense(units=4, activation=relu)(inputs)
outputs = Dense(units=2, activation=softmax)(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    loss=categorical_crossentropy,
    optimizer=Adam()
)
