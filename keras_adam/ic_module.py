# Kerasを使った分類
# https://qiita.com/neet-AI/items/2b3d7f743e4d6c6d8e10

import glob  # ファイル読み込み
import numpy as np  # 行列演算
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.preprocessing.image import random_rotation, random_shift, random_zoom
# ↑前処理
from keras.layers.convolutional import Conv2D
# ↑畳み込みのフォーマット
from keras.layers.pooling import MaxPooling2D
# ↑プーリング層（物体が画像のどの位置にあるかを無視する）
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential, model_from_json
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import np_utils  # 自然数をワンホットベクトルにするため

# 共通パラメータ
FileNames = ["img1.npy", "img2.npy", "img3.npy"]
ClassNames = ["ant", "butterfly", "lotus"]
hw = {"height": 32, "width": 32}

####
# 画像を読み込む前処理
####
def PreProcess(dirname, filename, var_amount=3):
    """
    dirnameファイル以下の画像(.jpg)を全て読み、
    画像を回転させてデータをvar_amount倍にする。
    :param dirname:
    :param filename:
    :param var_amount:
    :return:
    """

    num = 0  # ファイル数のカウンタ
    arrlist = []  # 画像ファイルの箱

    files = glob.glob(dirname + "\\*.jpg")  # ファイル名の抽出

    for imgfile in files:
        print(imgfile)
        img = load_img(imgfile,
                       target_size=(hw["height"], hw["width"])
                       )
        array = img_to_array(img) / 255  # 0~1の間に正規化
        arrlist.append(array)  # numpy型のリストに追加
        for i in range(var_amount-1):
            arr2 = array
            arr2 = random_rotation(arr2, rg=360)  # ランダム回転
            arrlist.append(arr2)
        num += 1

    nplist = np.array(arrlist)
    np.save(filename, nplist)
    print(">> " + dirname + "から" + str(num) + "個のファイル読み込みに成功")

####
# モデルの構築(CNN)
####
def BuildCNN(ipshape=(32, 32, 3), num_classes=3):
    """

    :param ipshape:
    :param num_classes:
    :return:
    """
    model = Sequential()  # 単純なモデル

    model.add(Conv2D(24, 3, padding='same', input_shape=ipshape))
    # 3*3のフィルタを使って24回畳み込みを行う、paddingは画像周りを0で囲みサイズを維持する
    model.add(Activation('relu'))

    model.add(Conv2D(48, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 2*2サイズの領域で最大のものをとり、並べる
    model.add(Dropout(0.5))
    # 50%を消す

    model.add(Conv2D(96, 3, padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(96, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))  # 要素128個の1次元配列
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))  # 読み込んだフォルダの数にする
    model.add(Activation('softmax'))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model

####
# 学習
####
def Learning(tsnum=30, nb_epoch=50, batch_size=8, learn_schedule=0.9):
    """

    :param tsnum:
    :param nb_epoch:
    :param batch_size:
    :param learn_schedule:
    :return:
    """
    # データの整理
    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []
    target = 0
    for filename in FileNames:
        data = np.load(filename)  # 画像のnumpyデータの読み込み
        trnum = data.shape[0] - tsnum
        x_train_list += [data[i] for i in range(trnum)]  # 画像データ
        y_train_list += [target] * trnum                 # 分類番号
        x_test_list += [data[i] for i in range(trnum, trnum+tsnum)]
        y_test_list += [target] * tsnum
        target += 1
    x_train = np.array(x_train_list + x_test_list)  # 連結
    y_train = np.array(y_train_list + y_test_list)
    print(">> 学習サンプル数：　", x_train.shape)
    y_train = np_utils.to_categorical(y_train, target)
    # 自然数をワンホットベクトル
    valrate = tsnum * target * 1.0 / x_train.shape[0]

    # 学習率の変更関数
    # 学習が進むにつれ重みを収束させる
    class Schedule(object):
        def __init__(self, init=0.001):
            self.init = init

        def __call__(self, epoch):
            lr = self.init
            for i in range(1, epoch+1):
                lr *= learn_schedule  # 学習の引数
            return lr

    def get_schedule_func(init):
        return Schedule(init)

    # 学習準備
    lrs = LearningRateScheduler(get_schedule_func(0.001))
    mcp = ModelCheckpoint(filepath='best.hdf5',
                          monitor='val_loss',
                          verbose=1,
                          save_best_only=True,
                          mode='auto')  # 学習途中でval_lossが最も小さくなるたびに重みを保存する
    model = BuildCNN(ipshape=(x_train.shape[1],
                              x_train.shape[2],
                              x_train.shape[3]),
                     num_classes=target)

    # 学習開始
    print(">> 学習開始")
    hist = model.fit(x_train, y_train,
                     batch_size=batch_size,
                     verbose=1,
                     epochs=nb_epoch,
                     validation_split=valrate,
                     callbacks=[lrs, mcp])
    # 学習に使用するデータ、バッチ、エポック、精度確認用のデータの割合、学習中に利用する関数

    # 保存
    json_string = model.to_json()
    json_string += '##########' + str(ClassNames)
    open('model.json', 'w').write(json_string)
    model.save_weights('last.hdf5')
    # 学習モデルをjsonで保存

####
# 実行
####
def TestProcess(imgname):
    """

    :param imgname:
    :return:
    """
    # 読み込み
    modelname_text = open("model.json").read()
    json_strings = modelname_text.split('##########')
    textlist = json_strings[1].replace("[", "").replace("]", "").replace("\'", "").split()
    model = model_from_json(json_strings[0])
    model.load_weights("last.hdf5")  # best.hdf5 で損失最小のパラメータを使用
    img = load_img(imgname,
                   target_size=(hw["height"], hw["width"])
                   )
    TEST = img_to_array(img) / 255
    
    # 画像の分類
    pred = model.predict(np.array([TEST]), batch_size=1, verbose=0)
    print(">> 計算結果↓\n" + str(pred))
    print(">> この画像は 「" + textlist[np.argmax(pred)].replace(",", "") + "」　です")