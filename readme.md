
# LSGAN中の転置畳み込み層の可視化

LSGANがどう100次元のノイズからどう画像を作っているのか知りたくてカーネルと特徴マップを可視化した。
正直、転置畳み込み層の各カーネルみても、なんでこんなシンプルなカーネルを複数畳み込んでいくと絵になるのかは全く理解できなかった。

# 使用したLSGANのコード
-  [『PyTorchニューラルネットワーク実装ハンドブック』](-  [『PyTorchニューラルネットワーク実装ハンドブック』](https://github.com/miyamotok0105/pytorch_handbook)の6章のコードを使用


# 使い方
- ノートブックはここ: `notebook/my_lsgan_debug.ipynb`
    -  予め36epoch学習したGeneratorの重みを`./weights`に入れている。
- もともとのLSGANモデルの学習スクリプトは`$ python train_lsgan.py`


