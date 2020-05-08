# 別のところにある画像を、大きさをそろえてjpg_originフォルダに入れる
# 参考：https://webbibouroku.com/Blog/Article/sklearn-cluster-flag
# pilではなくcv2を使う：https://peaceandhilightandpython.hatenablog.com/entry/2015/12/23/214840

import os
import cv2

jpgDir = "../keras_adam/"
jpgFolder_for_search = ["ant", "butterfly", "lotus"]

for folder in jpgFolder_for_search:
    print("start convert : ", folder)
    number = 0
    for path in os.listdir(f'{jpgDir}{folder}'):
        img = cv2.imread(f'{jpgDir}{folder}/{path}')  # 透明度：アルファチャンネルは読んでいない
        img_resize = cv2.resize(img, (200, 100))  # 値は適当 TODO
        cv2.imwrite(f'./jpg_convert/{folder}{number}.jpg', img_resize)
        number += 1
    print("finish convert : ", folder, "  ", str(number+1), " files")
