from functions import *

import os
import glob
# import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

# from keras import backend as k
# k.set_image_dim_ordering('tf')


# 学習
def train(model_name, epochs, batch_size, length, margin, angles):
    x, y = [], []

    # EMNISTからランダムに読み込み
    for label, folder in enumerate(glob.glob('./data/train/emnist/train_*/')):
        files = os.listdir(folder)
        # selects = random.sample(files, 1920)
        for file in files: # selects:
            img = cv2.imread(folder + file)

            for angle in angles:
                rot = angle_changer(img=img, angle=angle)

                cut = rectangle_cutout(img=rot)
                data = training_resize(img=cut, length=length, margin=margin)

                x.append(data)
                y.append(label)

        print('EMNIST reading_character', chr(label + 97), len(files))

    # TheChars74Kから読み込み
    for label, folder in enumerate(glob.glob('./data/train/chars74k/Sample0*/')):
        files = os.listdir(folder)
        for file in files:
            img = cv2.imread(folder + file)
            shrink = cv2.resize(img, None, fx=0.2, fy=0.2)

            for angle in angles:
                rot = angle_changer(img=shrink, angle=angle)

                cut = rectangle_cutout(img=rot)
                data = training_resize(img=cut, length=length, margin=margin)

                x.append(data)
                y.append(label)

        print('TheChars74K reading_character', chr(label + 97))

    # TTFファイルから画像生成
    for label in range(26):
        alphabet = chr(label + 97)
        for ttf in glob.glob('./data/train/ttf/*__*.ttf'):
            font = ImageFont.truetype(ttf, 100)
            width, height = font.getsize(alphabet)

            back = Image.new('RGB', (width * 2, height * 2), (255, 255, 255))
            draw = ImageDraw.Draw(back)
            draw.text((width / 2, height / 2), alphabet, (0, 0, 0), font)

            img = np.asarray(back)

            for angle in angles:
                rot = angle_changer(img=img, angle=angle)

                cut = rectangle_cutout(img=rot)
                data = training_resize(img=cut, length=length, margin=margin)

                x.append(data)
                y.append(label)

        print('TTF reading_character', alphabet)

    x = np.array(x)
    y = np.array(y)

    data_num = x.shape[0]
    x = x.reshape((data_num, length, length, 1)).astype(np.float32) / 255
    y = np_utils.to_categorical(y, 26)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    # モデル構造
    model = Sequential()

    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
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
    model.add(Dense(26, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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
    train(model_name='test', epochs=100, batch_size=256, length=28, margin=2, angles=[0, 10, 20, 350, 340])
