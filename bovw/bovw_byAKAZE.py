# AKAZEを用いた画像分類
# Caltech101を上位フォルダに開いて全て読み込む。
# https://blanktar.jp/blog/2016/03/python-visual-words
# https://qiita.com/hitomatagi/items/883770046de5746a5deb
# ここが一番よさそう
# https://hazm.at/mox/machine-learning/computer-vision/recipes/similar-image-retrieval.html

import cv2
import os
import glob
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
import shutil
import re

jpgDir = "../101_ObjectCategories/"
to_save = "./jpg_bovw_AKAZE"  # 分類後のフォルダ
os.makedirs(f'{to_save}', exist_ok=True)
to_csvName = "jpg_bovw_AKAZE.csv"

# 特徴量集合
features = []
# AKAZE特徴量を出すクラス
akaze = cv2.AKAZE_create()
# 直下フォルダの数：画像の種類
minisize = 20
count = 0
num_of_object = len(os.listdir(f'{jpgDir}')) - len(glob.glob(f'{jpgDir}/*.*'))
# 変換する画像の大きさ
size = (320, 240)
# bow分類器
if num_of_object > minisize:
    bowTrainer = cv2.BOWKMeansTrainer(minisize)
else:
    bowTrainer = cv2.BOWKMeansTrainer(num_of_object)
# Caltech101の画像を読み込んでサイズを変えて特徴量をとる
for folder in os.listdir(f'{jpgDir}'):
    print("load : ", folder)
    if count > minisize:
        break
    for path in os.listdir(f'{jpgDir}/{folder}'):
        print(f'{jpgDir}{folder}/{path}')
        img = cv2.imread(f'{jpgDir}{folder}/{path}', 0)  # 画像の読み込み, グレスケ
        # サイズのリサイズを行う。これをしないと特徴量抽出に失敗する。
        if img.shape[0] > img.shape[1]:
            img = cv2.resize(img, (size[1], size[1] * img.shape[0] // img.shape[1]))
        else:
            img = cv2.resize(img, (size[0] * img.shape[1] // img.shape[0], size[0]))
        print(img.size)
        keyPoints, descriptors = akaze.detectAndCompute(img, None)
        print(descriptors.astype)  # 56でエラー
        descriptors = descriptors.astype(np.float32)
        # detectAC, keyPoints, descriptorsで返り値を渡してくる
        bowTrainer.add(descriptors)
    count += 1

# 特徴ベクトルを分類
centroid = bowTrainer.cluster()
# 訓練完了
print("特徴量の抽出完了")

# テスト
print("test start")
# KNNで総当たりマッチング
matcher = cv2.BFMatcher()
# Bag Of Visual Words抽出器
bowExtractor = cv2.BOWImgDescriptorExtractor(akaze, matcher)
# トレーニング結果をセット
bowExtractor.setVocabulary(centroid)

# 正しく学習できたか検証する
for label in range(num_of_object):
    os.makedirs(f'{to_save}/{label}', exist_ok=True)

cols = ['path', 'folder', 'label']
df = pd.DataFrame(columns=cols)
count = 0
for folder in os.listdir(f'{jpgDir}'):
    print("load : ", folder)
    if count > minisize:
        break
    for path in os.listdir(f'{jpgDir}{folder}'):
#        print(f'{jpgDir}{folder}/{path}')
        img = cv2.imread(f'{jpgDir}{folder}/{path}', 0)  # 画像の読み込み, グレスケ
        # 特徴点と特徴ベクトルを計算
        keyPoints, descriptor = akaze.detectAndCompute(img, None)
#        descriptor = descriptor.astype(np.float32)
        # Bag Of Visual Wordsの計算
        bowDescriptor = bowExtractor.compute(img, keyPoints)

        label = folder
        shutil.copyfile(f'{jpgDir}{folder}/{path}',
                        f'{to_save}/{label}/{path}')
        print([label, path])

        # 分類結果のみを出力
        record = pd.Series([path, re.sub("\d*.jpg", "", path), label.copy()], index=df.columns)
        df = df.append(record, ignore_index=True)  # 参照なので代入しないといけない。遅いらしい。
    count += 1

df.to_csv(to_csvName)

# ==========================
#
# print(num_of_object, " : 写真の物体の種類")
# visual_words = MiniBatchKMeans(n_clusters=num_of_object*3,
#                                verbose=True
#                                ).fit(features).cluster_centers_
# # featuresで取り出した特徴量の重心をミニバッチにチャンクして計算し、クラスターの中心座標を返す
#
# # ヒストグラムの作成
# vector = np.zeros(len(visual_words))
# for f in features:
#     vector[((visual_words-f)**2).sum(axis=1).argmin()] += 1
#     # クラスター中心とのユークリッド距離が最小になる特徴量の位置に1を足す
# print(vector)
