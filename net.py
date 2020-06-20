# -*- coding: utf-8 -*-
import torch.nn as nn


# def weights_init(m):
#     """
#     ニューラルネットワークの重みを初期化する。作成したインスタンスに対しapplyメソッドで適用する
#     :param m: ニューラルネットワークを構成する層
#     """
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:            # 畳み込み層の場合
#         m.weight.data.normal_(0.0, 0.02)
#         m.bias.data.fill_(0)
#     elif classname.find('Linear') != -1:        # 全結合層の場合
#         m.weight.data.normal_(0.0, 0.02)
#         m.bias.data.fill_(0)
#     elif classname.find('BatchNorm') != -1:     # バッチノーマライゼーションの場合
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)


def weights_init(m):
    """NNの重みを初期化する。作成したインスタンスに対しapplyメソッドで適用する。
    m: NNを構成する層
    """

    classname = m.__class__.__name__

    # 畳み込み層の場合
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    # 全結合層の場合
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    # BNレイヤーの場合
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    """生成器Gのクラス
    """

    def __init__(self, nz=100, nch_g=64, nch=3):
        """
        nz: 入力次元
        nch_g: 最終層の入力チャネル数
        nch: 出力のチャネル数
        """
        super(Generator, self).__init__()

        self.layers = nn.ModuleList([
            #  L0: (N, z, 1, 1) → (N, C*8, 4, 4)
            nn.Sequential(
                nn.ConvTranspose2d(nz,  # 入力画像のチャンネル数
                                   nch_g * 8,  # 出力されるチャンネル数
                                   4,  # 畳み込みカーネルのサイズ
                                   1,  # ストライド
                                   0),  # パディングの値
                nn.BatchNorm2d(nch_g * 8),
                nn.ReLU()
            ),
            #  L1: (N, C*8, 4, 4) → (N, C*4, 8, 8)
            nn.Sequential(
                nn.ConvTranspose2d(nch_g * 8,  # 入力画像のチャンネル数
                                   nch_g * 4,  # 出力されるチャンネル数
                                   4,  # 畳み込みカーネルのサイズ
                                   2,  # ストライド
                                   1),  # パディングの値
                nn.BatchNorm2d(nch_g * 4),
                nn.ReLU()
            ),
            # L2
            nn.Sequential(
                nn.ConvTranspose2d(nch_g * 4,  # 入力画像のチャンネル数
                                   nch_g * 2,  # 出力されるチャンネル数
                                   4,  # 畳み込みカーネルのサイズ
                                   2,  # ストライド
                                   1),  # パディングの値
                nn.BatchNorm2d(nch_g * 2),
                nn.ReLU()
            ),
            #  L3(N, C*4, 8, 8) → (N, C*2, 16, 16)
            nn.Sequential(
                nn.ConvTranspose2d(nch_g * 2,  # 入力画像のチャンネル数
                                   nch_g,  # 出力されるチャンネル数
                                   4,  # 畳み込みカーネルのサイズ
                                   2,  # ストライド
                                   1),  # パディングの値
                nn.BatchNorm2d(nch_g),
                nn.ReLU()
            ),
            #  L4(N, C*2, 16, 16) → (N, C, 32, 32)
            nn.Sequential(
                nn.ConvTranspose2d(nch_g,  # 入力画像のチャンネル数
                                   nch,  # 出力されるチャンネル数
                                   4,  # 畳み込みカーネルのサイズ
                                   2,  # ストライド
                                   1),  # パディングの値
                nn.Tanh()
            )  # (N, C, 32, 32) -> (N, out_ch, 64, 64)
        ])

    def forward(self, z):
        """
        """
        for layer in self.layers:
            z = layer(z)
        return z


class Discriminator(nn.Module):
    """
    識別器Dのクラス
    """

    def __init__(self, nch=3, nch_d=64):
        """
        """
        super(Discriminator, self).__init__()

        # NNの構造を定義
        self.layers = nn.ModuleList([
            # a
            nn.Sequential(
                nn.Conv2d(nch, nch_d, 4, 2, 1),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            # b
            nn.Sequential(
                nn.Conv2d(nch_d, nch_d * 2, 4, 2, 1),
                nn.BatchNorm2d(nch_d * 2),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            # c
            nn.Sequential(
                nn.Conv2d(nch_d * 2, nch_d * 4, 4, 2, 1),
                nn.BatchNorm2d(nch_d * 4),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            # d
            nn.Sequential(
                nn.Conv2d(nch_d * 4, nch_d * 8, 4, 2, 1),
                nn.BatchNorm2d(nch_d * 8),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            # e
            nn.Conv2d(nch_d * 8, 1, 4, 1, 0),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        print(x.shape)
        return x.squeeze()  # Tensorの形を(N)にする
