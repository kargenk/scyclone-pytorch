import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LogMelspDataset
from logger import Logger
from model import Scyclone
from utils import worker_init_fn


def log(logger: object, log_dict: dict, i: int):
    """
    Tensor Boardに各損失と訓練状況のログを吐き出す．

    Args:
        logger (object): TensorBoardの自作ラッパクラス
        log_dict (dict): 損失関数の辞書
        i (int): エポック数
    """
    # 損失関数のログ
    for tag, value in log_dict.items():
        logger.scalar_summary(tag, value, i)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # データローダーの作成
    jvs001_to_jvs015_dataset = LogMelspDataset(source_dir='data/processed_logmel_unique_win1024shift256/jvs001',
                                               target_dir='data/processed_logmel_unique_win1024shift256/jvs015')
    print(f'jvs001 to jvs015: {len(jvs001_to_jvs015_dataset)}')

    # データローダーの作成
    train_loader = DataLoader(jvs001_to_jvs015_dataset,
                              batch_size=64,    # バッチサイズ
                              shuffle=True,     # データシャッフル
                              num_workers=2,    # 高速化
                              pin_memory=True,  # 高速化
                              worker_init_fn=worker_init_fn)

    print(iter(train_loader).next()[0].shape)
    print(iter(train_loader).next()[1].shape)
    print(len(train_loader))

    # モデル，最適化アルゴリズム，スケジューラ，TensorBoardの設定
    model = Scyclone()
    model.configure_optimizers()
    logger = Logger('outputs/logs')

    # 訓練の実行
    epoch = 100000
    for epoch in tqdm(range(1, epoch + 1)):
        for batch in train_loader:
            src, trg = batch
            batch = (src.to(device), trg.to(device))

            # Discriminatorの訓練
            model.optim_D.zero_grad()
            out_d_dict = model.train_d(batch)
            out_d_dict['loss'].backward()
            model.optim_D.step()
            # print(f'epoch:{epoch}, d_lr:{scheduler_D.get_last_lr()[0]}')
            out_d_dict['log']['Discriminator/LearningRate'] = model.scheduler_D.get_last_lr()[0]
            model.scheduler_D.step()
            log(logger, out_d_dict['log'], epoch)

            # Generatorの訓練
            model.optim_G.zero_grad()
            out_g_dict = model.train_g(batch)
            out_g_dict['loss'].backward()
            model.optim_G.step()
            # print(f'epoch:{epoch}, g_lr:{scheduler_G.get_last_lr()[0]}')
            out_g_dict['log']['Generator/LearningRate'] = model.scheduler_G.get_last_lr()[0]
            model.scheduler_G.step()
            log(logger, out_g_dict['log'], epoch)

            # モデルとOptimizerの保存
            if epoch % 10000 == 0:
                model.save_models(epoch, 'models')
                model.save_optims(epoch, 'optims')
