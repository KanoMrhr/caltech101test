from keras_adam import ic_module as ic
import os.path as op

i = 0
for filename in ic.FileNames:
    # ディレクトリ名を入力
    while True:
        dirname = input(">> 「" + ic.ClassNames[i] + "」の画像のあるディレクトリ：")
        if op.isdir(dirname):
            break
        print(">> そのディレクトリは存在しません")

    # 関数の実行
    ic.PreProcess(dirname,
                  filename,
                  var_amount=3)
    i += 1
