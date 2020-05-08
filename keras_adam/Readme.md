# このフォルダについて
kerasとadamを使って、画像のクラス分類を行うプログラムが入っている。
参考サイト：https://qiita.com/neet-AI/items/2b3d7f743e4d6c6d8e10

# 各ファイルについて
img~.npyは読み込んだ画像をnumpy形式で保存したもの。

ant, butterfly, lotusフォルダはCaltech101の各画像である。

model.jsonが学習モデルである。

.hdf5の拡張子のファイルは重み？を保存していた気がする。

ic_module.pyにメインプログラムが入っており、
他の.pyプログラムはic_moduleの各メソッドを用いて動く。
