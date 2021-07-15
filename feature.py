from functions import *

import os
import glob
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ["large", "small"]
DATA_SCALE = "small"


def get_feature(angles, length, margin):

    x, y = [], []

    # EMNISTからランダムに読み込み
    for label, folder in enumerate(glob.glob('./data/train/emnist/train_*/')):
        files = os.listdir(folder)

        if DATA_SCALE == "large":
            selects = random.sample(files, len(files))

        if DATA_SCALE == "small":
            selects = random.sample(files, 1920)

        for file in selects:
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

    features = []

    for _y, _x in zip(y, x):

        feature = flatten_with_any_depth([_y, _x.tolist()])
        features.append(feature)

    print("\nWriting features to CSV...")
    write_csv("./data/train.csv", features)


if __name__ == '__main__':
    get_feature(angles=[0, 10, 20, 350, 340], length=28, margin=2)
