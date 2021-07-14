import glob
import csv


# ユーザ・インタフェースごとに誤認識をカウント
def miss_count():
    for model in ['CNN_alphabet_181112', 'CNN_alphabet_190207']:
        folder = './誤認識リスト_{}'.format(model)

        log = open('{}/誤認識数_{}.csv'.format(folder, model), 'w', newline='')
        writer = csv.writer(log)
        writer.writerow([model])

        for frame in ['eleven', 'none', 'two']:
            for user in ['a-1', 'a-2', 'b-1', 'b-2', 'c-1', 'c-2', 'd-1', 'd-2', 'e-1', 'e-2', 'f-1', 'f-2']:
                target = '{}_{}'.format(user, frame)

                count = 0
                for _ in glob.glob('{}/*/*_{}_*.png'.format(folder, target)):
                    count += 1

                writer.writerow([user, frame, count])

        log.close()


if __name__ == '__main__':
    miss_count()
