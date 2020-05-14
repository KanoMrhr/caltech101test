# AKAZEを用いた画像分類
# Caltech101を上位フォルダに開いて全て読み込む。
# https://blanktar.jp/blog/2016/03/python-visual-words
# https://qiita.com/hitomatagi/items/883770046de5746a5deb

import cv2
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans

jpgDir = "../101_ObjectCategories/"
# to_save = "./jpg_convert_Canny"
# os.makedirs(f'{to_save}', exist_ok=True)

features = []  # 特徴量集合
akaze = cv2.AKAZE_create()  # AKAZE特徴量を出すクラス
num_of_object = 0
for folder in os.listdir(f'{jpgDir}'):
    print("load : ", folder)
    folder = "dolphin"
    for path in os.listdir(f'{jpgDir}/{folder}'):
        print(f'{jpgDir}{folder}/{path}')
        img = cv2.imread(f'{jpgDir}{folder}/{path}')  # 画像の読み込み, グレスケ
        keypoints, desctriptors = akaze.detectAndCompute(img, None)
        # detectACば, keypoints, descriptorsで返り値を渡してくる
#        print(desctriptors.shape)
#        print(desctriptors.astype)
        features.extend(desctriptors.astype(np.float32))
    num_of_object += 1

print(num_of_object, " : 写真の物体の種類")
visual_words = MiniBatchKMeans(n_clusters=num_of_object*3
                               ).fit(features).cluster_centers_
# featuresで取り出した特徴量の重心をミニバッチにチャンクして計算し、クラスターの中心座標を返す

# ヒストグラムの作成
vector = np.zeros(len(visual_words))
for f in features:
    vector[((visual_words-f)**2).sum(axis=1).argmin()] += 1
    # クラスター中心とのユークリッド距離が最小になる特徴量の位置に1を足す
print(vector)
