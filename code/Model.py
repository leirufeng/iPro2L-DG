from keras.layers import Input, Flatten
from keras.models import Model
from keras.optimizer_v2.adam import Adam

from DenseNet import *
from GAM import *



def build_model(windows=5, denseblocks=5, layers=3, filters=96,
                growth_rate=32, dropout_rate=0.5, weight_decay=1e-4):
    input_1 = Input(shape=(windows, 81, 1))

    for i in range(denseblocks - 1):
        # Add denseblock
        x_1, filters_1 = denseblock(input_1, layers=layers,
                                    filters=filters, growth_rate=growth_rate,
                                    dropout_rate=dropout_rate, weight_decay=weight_decay)
        # Add BatchNormalization
        x_1 = BatchNormalization(axis=-1)(x_1)

        # Add transition
        x_1 = transition(x_1, filters=filters_1,
                         dropout_rate=dropout_rate, weight_decay=weight_decay)
    # The last denseblock
    # Add denseblock
    x_1, filters_1 = denseblock(x_1, layers=layers,
                                filters=filters, growth_rate=growth_rate,
                                dropout_rate=dropout_rate, weight_decay=weight_decay)
    # Add BatchNormalization
    x_1 = BatchNormalization(axis=-1)(x_1)
    # # Adding an attention module layer

    x_1 = GAMAttention(288)(x_1)

    # # Pooling
    x_1 = AveragePooling2D(pool_size=(2, 12), strides=(1, 1))(x_1)

    # Flatten
    x = Flatten()(x_1)

    # MLP

    x = Dense(units=2, activation="softmax", use_bias=False,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x)

    inputs = [input_1]
    outputs = [x]

    model = Model(inputs=inputs, outputs=outputs, name="iPro2L")

    optimizer = Adam(lr=2.2e-4, epsilon=1e-8)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
