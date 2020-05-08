# jpg_converterによって整理された画像をjpg_groupフォルダに分類する
# Spectral-Clusteringにある最近傍法で分類する
# http://neuro-educator.com/ml10/

import os
import numpy as np
import shutil  # ファイル操作用
from skimage import io  # 手もとの画像を使うとき？はdataではできなさそうだった。
from sklearn.cluster import SpectralClustering

import pandas as pd  # ログをcsvでも保管しておく
# https://pythondatascience.plavox.info/pandas/%E3%83%87%E3%83%BC%E3%82%BF%E3%83%95%E3%83%AC%E3%83%BC%E3%83%A0%E3%82%92%E5%87%BA%E5%8A%9B%E3%81%99%E3%82%8B

assign_label = ['kmeans', 'discretize']  # kmeansと離散化
folder = "./jpg_convert"
to_save = ["./jpg_group2", "./jpg_group3"]   # 保存先のフォルダ（作成する）
to_csvName = ["jpg_group_result2", "jpg_group_result3"]

print("Start")
# 1. 画像を読み込み、3次元→2次元配列データに変換
jpgs = np.array([
    io.imread(f'{folder}/{path}') for path in os.listdir(f'{folder}')
])
jpgs = jpgs.reshape(len(jpgs), -1).astype(np.float64)
# 次元を減らす(100, 200, 3)->(100, 600)

# 2. グループ分け学習
print("learning...")
for i in range(len(assign_label)):
    model = SpectralClustering(n_clusters=3,
                               affinity="nearest_neighbors",
                               assign_labels=assign_label[i]
                               ).fit(jpgs)
    labels = model.labels_  # 結果のラベル:整数値

    # 3. 結果
    print("Result...")
    cols = ['label', 'path']
    df = pd.DataFrame(columns=cols)
    for label, path in zip(labels, os.listdir(f'{folder}')):
        # フォルダごとに画像を分類
        os.makedirs(f'{to_save[i]}/{label}', exist_ok=True)
        shutil.copyfile(f'{folder}/{path}',
                        f'{to_save[i]}/{label}/{path}')
        print([label, path])

        # 分類結果のみを出力
        record = pd.Series([label.copy(), path], index=df.columns)
        df = df.append(record, ignore_index=True)  # 参照なので代入しないといけない。遅いらしい。
    df.to_csv(to_csvName[i])


print("END")

