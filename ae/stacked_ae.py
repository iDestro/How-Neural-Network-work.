import tensorflow as tf
from tensorflow import keras


stacked_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation='selu'),
    # 式子比较复杂的激活函数
    keras.layers.Dense(30, activation='selu')
])

stacked_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation='selu'),
    keras.layers.Dense(28*28, activation='selu', kernel_initializer='g'),
    keras.layers.Reshape([28, 28])
])

stacked_ae = keras.models.Sequential([stacked_encoder, stacked_decoder])
stacked_ae.compile(loss='binary_crossentropy',
                   optimizer=keras.optimizers.SGD(lr=1.5))

# history = stacked_ae.fit(X, Y, epochs=10, validation_data=[X_valid, Y_valid])

