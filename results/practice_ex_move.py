import glob
import os
import shutil


# 練習と例題の画像を移動
def practice_ex_move():
    for user in glob.glob('./*-*'):

        path = '{}/practice_ex'.format(user)
        os.makedirs(path, exist_ok=True)

        for target in ['practice', 'ex']:
            for file in glob.glob('{}/*_{}_*.png'.format(user, target)):
                shutil.move(file, path)
                print(file)


if __name__ == '__main__':
    practice_ex_move()
