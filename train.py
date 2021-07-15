from functions import *

import os
import numpy as np

import keras
from keras.utils import np_utils
from keras.callbacks import CSVLogger, ModelCheckpoint # , EarlyStopping
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE # pip install imbalanced-learn

# from keras import backend as k
# k.set_image_dim_ordering('tf')

import cnn

# model_type = ["deep": "deep_cnn", "standard": "standard_model"]
MODEL_TYPE = "deep"


# 学習
def train(model_name, epochs, batch_size, length):

    print("Reading features from CSV...")
    x, y = load_data(file_path="./data/train.csv")

    print("\nOver sampling by SMOTE.\n")
    smote = SMOTE(random_state=42)
    x, y = smote.fit_resample(x, y)

    for i in range(26):
        print("Data number resampled => " + chr(i + 97) + ": " + str(np.sum(y == i)))

    data_num = x.shape[0]
    x = x.reshape((data_num, length, length, 1)).astype(np.float32) / 255
    y = np_utils.to_categorical(y, 26)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    if MODEL_TYPE == "standard":
        model = cnn.create_standard_model(length=length, y_len=26)

    if MODEL_TYPE == "deep":
        model = cnn.create_deep_model(length=length, y_len=26, learn_rate=0.0001)

    # 学習結果を保存
    save_folder = './model/{}/'.format(model_name)
    os.makedirs(save_folder, exist_ok=True)

    csv_logger = CSVLogger(save_folder + 'log_acc_loss.csv', append=False)
    model_checkpoint = ModelCheckpoint(save_folder + 'CNN_model_epoch_{epoch:03d}.h5', monitor='val_loss', verbose=1,
                                       save_best_only=False, save_weights_only=False, mode='min', period=0)
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='auto')

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test),
                        verbose=1, callbacks=[csv_logger, model_checkpoint]) # , early_stopping])
    plot_result(save_folder, history)

    for i in model.layers:
        if type(i) is keras.layers.Dropout:
            model.layers.remove(i)

    model.save('{}/CNN_model_{}.h5'.format(save_folder, model_name))

    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test Loss     :', score[0])
    print('Test Accuracy :', score[1])


if __name__ == '__main__':
    train(model_name='temp', epochs=100, batch_size=256, length=28)
