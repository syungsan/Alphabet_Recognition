import glob
import cv2
import os


# 切り出し前の画像抽出
def before_trim_move(width, height):
    count = 0
    for file in glob.glob('./*-*/_rect/*.png'):
        if file.find('practice') == -1 and file.find('ex') == -1:
            if count not in [52, 319, 321, 757, 799, 804, 813, 818, 827, 832, 1213, 1388, 1430, 1436, 1461, 1462, 1466, 1472]:

                # twoの場合の横幅調整
                img = cv2.imread(file)
                new_width = int(img.shape[1]/85 * width)
                resize = cv2.resize(img, (new_width, height))

                user = os.path.basename(os.path.dirname(os.path.dirname(file)))
                name = os.path.basename(file)[5:]

                folder = './before/{}'.format(user)
                os.makedirs(folder, exist_ok=True)

                save = '{}/{}_{}'.format(folder, user, name)
                cv2.imwrite(save, resize)

            count += 1


if __name__ == '__main__':
    before_trim_move(width=65, height=120)
