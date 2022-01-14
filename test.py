import os
import csv
import glob
import numpy as np
from PIL import Image
from keras.models import load_model
import fnmatch

THINNING_OUT_NUMBER = 10


# 文字認識検証,ユーザごとにフォルダ分けをしておく
def character_recognition(recognition_folder, model_folder, error_index):

    # 1文字ずつ順に正解文字を保存
    count = 0
    corrects = []

    for user in ['a-1', 'a-2', 'b-1', 'b-2', 'c-1', 'c-2', 'd-1', 'd-2', 'e-1', 'e-2', 'f-1', 'f-2']:
        for frame in ['eleven', 'none', 'two']:
            for index, word in enumerate(['size', 'quick', 'maybe', 'five', 'light', 'power', 'next', 'study', 'enjoy']):

                # twoにおける1文字不足分の削除
                for lack in [['a-2', 2], ['a-2', 3], ['b-1', 5], ['d-1', 0], ['d-1', 2], ['f-2', 6]]:
                    if user == lack[0] and frame == 'two' and index == lack[1]:
                        word = word[:-1]

                # 不正解単語の置き換え
                for incorrect in [['eleven', 3, 'take'], ['eleven', 8, 'indwry'], ['none', 8, 'introy'], ['two', 8, 'intory']]:
                    if user == 'e-2' and frame == incorrect[0] and index == incorrect[1]:
                        word = incorrect[2]

                for character in word:

                    # エラー文字分の除外
                    if count not in error_index:
                        corrects.extend(character)

                    count += 1

    # 文字認識検証
    result_folder = './evaluation/検証結果_{}_{}'.format(os.path.basename(recognition_folder), model_folder)
    os.makedirs(result_folder, exist_ok=True)

    all_log = open('{}/誤認識まとめ.csv'.format(result_folder), 'w', newline='')
    all_writer = csv.writer(all_log)
    all_writer.writerow(['認識フォルダ', 'モデル', '誤認識数', '認識率'])

    # for model_path in glob.glob('./model/{}/*.h5'.format(model_folder)):
    final_model_paths = glob.glob('./model/{}/*_{}.h5'.format(model_folder, model_folder))
    pre_process_model_paths = glob.glob('./model/{}/*_epoch_*.h5'.format(model_folder))

    model_paths = []
    model_paths.extend(final_model_paths)

    for index in range(THINNING_OUT_NUMBER, int(len(pre_process_model_paths)), THINNING_OUT_NUMBER):
        model_paths.extend(fnmatch.filter(pre_process_model_paths, "*_{}.h5".format("{0:03d}".format(index))))

    for model_path in model_paths:
        model = load_model(model_path)

        model_name, _ = os.path.splitext(os.path.basename(model_path))
        os.makedirs('{}/{}'.format(result_folder, model_name), exist_ok=True)

        rows = []
        for index, file in enumerate(glob.glob('{}/*-*/*.png'.format(recognition_folder))):
            img = Image.open(file).convert('L').resize((28, 28))

            x = np.array([np.asarray(img)])
            x = x.reshape((1, 28, 28, 1)).astype(np.float32) / 255

            label = model.predict(x)
            alphabet = chr(label.argmax() + 97)
            probability = label.ravel().tolist()

            rows.append([file, corrects[index], alphabet] + probability)

        # 誤認識率_記録
        prob_log = open('{}/{}/誤認識率.csv'.format(result_folder, model_name), 'w', newline='')
        prob_writer = csv.writer(prob_log)
        prob_writer.writerow(['文字画像', '正解文字', '認識結果',
                              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                              'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])

        # 文字画像と集計
        misses = []
        miss_corrects = []
        for row in rows:
            img_path, correct, incorrect = row[0], row[1], row[2]
            user = os.path.basename(os.path.dirname(img_path))
            png = os.path.basename(img_path)

            if not correct == incorrect:
                prob_writer.writerow(row)

                img = Image.open(img_path)
                img.save('{}/{}/{}-{}__{}__{}'.format(result_folder, model_name, correct, incorrect, user, png))

                misses.extend(['{}→{}'.format(correct, incorrect)])
                miss_corrects.extend(correct)

        prob_log.close()

        # 誤認識集計_記録
        list_log = open('{}/{}/誤認識集計.csv'.format(result_folder, model_name), 'w', newline='')
        list_writer = csv.writer(list_log)

        # 誤認識の種類
        dictionary = {}
        for miss in misses:
            dictionary[miss] = misses.count(miss)

        list_writer.writerow(['誤認識種類', '誤認識数'])
        for key, value in sorted(dictionary.items(), key=lambda v: v[1], reverse=True):
            list_writer.writerow([key, value])
        list_writer.writerow(['合計', len(misses)])

        # 各文字の認識率
        list_writer.writerow(['文字', '認識率'])
        for code in range(97, 123):
            alphabet = chr(code)

            character_sum = corrects.count(alphabet)
            miss_sum = miss_corrects.count(alphabet)
            probability = '{}%'.format(round(100 * (character_sum - miss_sum) / character_sum, 1))

            list_writer.writerow([alphabet, probability])

        all_probability = '{}%'.format(round(100 * (len(rows) - len(miss_corrects)) / len(rows), 1))
        list_writer.writerow(['全体', all_probability])

        list_log.close()

        # まとめ_記録
        all_writer.writerow([recognition_folder, model_name, len(misses), all_probability])
        print('Test End : ', recognition_folder, model_name)

    all_log.close()


if __name__ == '__main__':

    # # 卒論
    # thesis_list = [52, 319, 321, 757, 799, 804, 813, 818, 827, 832, 1388, 1430, 1436, 1461, 1462, 1466, 1472]
    # character_recognition(recognition_folder='results', model_folder='181112', error_index=thesis_list)

    # 新規試行
    thesis_list = [52, 319, 321, 757, 799, 804, 813, 818, 827, 832, 1388, 1430, 1436, 1461, 1462, 1466, 1472]
    character_recognition(recognition_folder='results', model_folder='data-small-emnist_non-expansioned_type-shallow', error_index=thesis_list)

    # 切り出し調整後
    # add_list = [52, 319, 321, 757, 799, 804, 813, 818, 827, 832, 1213, 1388, 1430, 1436, 1461, 1462, 1466, 1472]
    # character_recognition(recognition_folder='data/test/margin_2', model_folder='alphabet_X', error_index=add_list)

    add_list = [52, 319, 321, 757, 799, 804, 813, 818, 827, 832, 1213, 1388, 1430, 1436, 1461, 1462, 1466, 1472]
    character_recognition(recognition_folder='data/test/margin_2', model_folder='data-small-emnist_non-expansioned_type-shallow', error_index=add_list)

    for a in glob.glob('results/*-*/.*'):
        os.remove(a)
        print(a)
