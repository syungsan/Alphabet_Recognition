from keras.models import Model, Sequential
from keras.layers.core import Dropout, Dense
from keras.layers import Input, GlobalAveragePooling2D, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D


def create_standard_model(length, y_len):

    # モデル構造
    model = Sequential()

    model.add(Conv2D(32, (5, 5), input_shape=(length, length, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(y_len, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def create_deep_model(length, y_len, learn_rate):

    inputs = Input(shape=(length, length, 1))

    x = Conv2D(64, (3, 3), padding="SAME", activation="relu")(inputs)
    x = Conv2D(64, (3, 3), padding="SAME", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding="SAME", activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), padding="SAME", activation="relu")(x)
    x = Conv2D(128, (3, 3), padding="SAME", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding="SAME", activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.25)(x)

    x = Conv2D(256, (3, 3), padding="SAME", activation="relu")(x)
    x = Conv2D(256, (3, 3), padding="SAME", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding="SAME", activation="relu")(x)

    x = Conv2D(256, (3, 3), padding="SAME", activation="relu")(x)
    x = Conv2D(256, (3, 3), padding="SAME", activation="relu")(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, (3, 3), padding="SAME", activation="relu")(x)
    x = Conv2D(512, (3, 3), padding="SAME", activation="relu")(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    y = Dense(units=y_len, activation="softmax")(x)

    model = Model(inputs, y)

    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=learn_rate, amsgrad=True), metrics=["accuracy"])
    model.summary()

    return model
