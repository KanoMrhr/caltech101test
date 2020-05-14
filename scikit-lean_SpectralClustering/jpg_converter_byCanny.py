import cv2
import os

jpgDir = "../keras_adam/"
jpgFolder_for_search = ["ant", "butterfly", "lotus"]
to_save = "./jpg_convert_Canny"
os.makedirs(f'{to_save}', exist_ok=True)

for folder in jpgFolder_for_search:
    print("start convert : ", folder)
    number = 0
    for path in os.listdir(f'{jpgDir}{folder}'):
        gray = cv2.imread(f'{jpgDir}{folder}/{path}',  # 透明度：アルファチャンネルは読んでいない
                         cv2.IMREAD_GRAYSCALE)  # grayscale化する
        edges = cv2.Canny(gray,
                          100,
                          200,
                          apertureSize=3,  # ソーベルフィルタのサイズ
                          L2gradient=True)  # 勾配強度を計算するための式を精度の良いものにするかどうか
        # cv2.imshow("gray", gray)
        # cv2.imshow("edges", edges)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # break
        img_resize = cv2.resize(edges, (200, 100))  # 値は適当 TODO
        cv2.imwrite(f'{to_save}/{folder}{number}.jpg', img_resize)
        number += 1
    # break
    print("finish convert : ", folder, "  ", str(number+1), " files")

