import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


# 回転拡張
def angle_changer(img, angle):
    if angle != 0:
        height, width, _ = img.shape

        rot_mat = cv2.getRotationMatrix2D((width/2, height/2), angle, scale=1.0)
        rot = cv2.warpAffine(img, rot_mat, (width, height), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))

    else:
        rot = img

    return rot


# 矩形で切り出し
def rectangle_cutout(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY)
    reverse = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(reverse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        ary = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            ary.extend([[x, y, x + w, y + h]])

        x1 = np.array(ary).min(axis=0)[0]
        y1 = np.array(ary).min(axis=0)[1]
        x2 = np.array(ary).max(axis=0)[2]
        y2 = np.array(ary).max(axis=0)[3]
        rect = img[y1:y2, x1:x2]

    else:
        rect = img

    return rect


# 正方形に合わせる
def training_resize(img, length, margin):
    front = Image.fromarray(img)

    width, height = front.size
    size = max(width, height) * length // (length - margin * 2)
    pos_x, pos_y = (size - width) // 2, (size - height) // 2

    back = Image.new(front.mode, (size, size), (255, 255, 255))
    back.paste(front, (pos_x, pos_y))
    
    train_img = back.convert('L').resize((length, length))
    data = np.asarray(train_img)

    return data


# 学習結果グラフ
def plot_result(log_path, history):
    plt.figure()
    plt.plot(history.history['accuracy'], label='acc', marker='.')
    plt.plot(history.history['val_accuracy'], label='val_acc', marker='.')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('{}/graph_acc.png'.format(log_path))
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'], label='loss', marker='.')
    plt.plot(history.history['val_loss'], label='val_loss', marker='.')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('{}/graph_loss.png'.format(log_path))
    plt.show()
