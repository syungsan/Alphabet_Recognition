#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_matching_point_number(query_img_path, map_img_path, is_show_fig=False):

    akaze = cv2.AKAZE_create()

    # gamma補正の関数
    gamma = 1.8
    gamma_cvt = np.zeros((256, 1), dtype='uint8')
    for i in range(256):
        gamma_cvt[i][0] = 255 * (float(i)/255) ** (1.0/gamma)

    # クエリ画像を読み込んで特徴量計算
    query_img = cv2.imread(query_img_path, 0)
    query_img = cv2.LUT(query_img, gamma_cvt)
    kp_query, des_query = akaze.detectAndCompute(query_img, None)

    # マップ画像を読み込んで特徴量計算
    map_img = cv2.imread(map_img_path, 0)
    kp_map, des_map = akaze.detectAndCompute(map_img, None)

    # 特徴量マッチング実行，k近傍法
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_query, des_map, k=2)

    # マッチング精度が高いもののみ抽出
    ratio = 0.8  # 重要パラメータ
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append([m])

    # 対応点が１個以下なら相対関係を求められないのでNoneを返す
    if len(good) <= 1:
        print("[error] can't detect matching feature point")
        return None

    # 精度が高かったもののうちスコアが高いものから指定個取り出す
    good = sorted(good, key=lambda x: x[0].distance)
    print("valid point number: ", len(good))  # これがあまりに多すぎたり少なすぎたりする場合はパラメータを変える

    if is_show_fig:
        point_num = 20  # 上位何個の点をマッチングポイントの描画に使うか
        if len(good) < point_num:
            point_num = len(good)  # もし20個なかったら全て使う

        # マッチング結果の描画
        result_img = cv2.drawMatchesKnn(query_img, kp_query, map_img, kp_map, good[:point_num], None, flags=0)
        img_matching = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_matching)
        plt.show()

    return len(good)


if __name__ == "__main__":

    matching_point_num = get_matching_point_number('./data/test_imgs/v1.png', './data/test_imgs/v2.png', is_show_fig=True)
    print(matching_point_num)
