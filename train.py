import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LogMelspDataset
from logger import Logger
from model import Scyclone
from utils import fix_seed, worker_init_fn


def loss_dic_update(batch_dict: dict, running_loss_dict: dict = {}) -> dict:
    """各lossをバッチで集計(総和)したものを返す.

    Args:
        batch_dict (dict): バッチごとの損失関数の辞書
        running_loss_dict (dict): バッチごとの値を加算して更新した損失関数の辞書

    Returns:
        dict: バッチごとの値を加算して更新した損失関数の辞書
    """
    for key, value in batch_dict['log'].items():
        if running_loss_dict.get(key) is None:
            running_loss_dict[key] = value.item()
        else:
            running_loss_dict[key] += value.item()
    return running_loss_dict


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
    fix_seed(42)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    exp_name = '0317_kiritan-no7_b1024_for24kHz_usepretrain'

    # コマンドライン引数の受け取り
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_iter', type=int, help='iteration of pretrained model.')
    parser.add_argument('--weights_dir', help='weights directory path')
    args = parser.parse_args()

    # データローダーの作成
    kiritan_to_no7_dataset = LogMelspDataset(source_dir='data/processed_logmel_kiritan-no7_24k/kiritan/train/',
                                             target_dir='data/processed_logmel_kiritan-no7_24k/no7/train/')
    print(f'kiritan to no7: {len(kiritan_to_no7_dataset)}')

    # データローダーの作成
    train_loader = DataLoader(kiritan_to_no7_dataset,
                              batch_size=1024,  # バッチサイズ
                              shuffle=True,     # データシャッフル
                              num_workers=2,    # 高速化
                              pin_memory=True,  # 高速化
                              worker_init_fn=worker_init_fn)

    print(iter(train_loader).next()[0].shape)
    print(f'iteration size: {len(train_loader)}')

    # モデル，最適化アルゴリズム，スケジューラ，TensorBoardの設定
    model = Scyclone()
    if args.resume_iter:
        model.restore_models(i=args.resume_iter, weights_dir=args.weights_dir)
    model.configure_optimizers()
    logger = Logger(os.path.join(exp_name, 'outputs', 'logs'))

    # 訓練の実行
    epoch = 300000
    for epoch in tqdm(range(1, epoch + 1)):
        running_dloss_dict = {}
        running_gloss_dict = {}
        for batch in train_loader:
            src, trg = batch
            batch = (src.to(device), trg.to(device))

            # Discriminatorの訓練
            model.optim_D.zero_grad()
            out_d_dict = model.train_d(batch)
            out_d_dict['loss'].backward()
            model.optim_D.step()

            # Generatorの訓練
            model.optim_G.zero_grad()
            out_g_dict = model.train_g(batch)
            out_g_dict['loss'].backward()
            model.optim_G.step()

            # 各損失ごとにバッチ間で総和を取って集計する
            running_dloss_dict = loss_dic_update(out_d_dict, running_dloss_dict)
            running_gloss_dict = loss_dic_update(out_g_dict, running_gloss_dict)

        # エポック内での平均値にならす
        for key in running_dloss_dict.keys():
            running_dloss_dict[key] /= len(train_loader)
        for key in running_gloss_dict.keys():
            running_gloss_dict[key] /= len(train_loader)

        # 学習率のスケジューラを更新してTensorBoardに出力
        running_dloss_dict['Discriminator/LearningRate'] = model.scheduler_D.get_last_lr()[0]
        running_gloss_dict['Generator/LearningRate'] = model.scheduler_G.get_last_lr()[0]
        model.scheduler_D.step()
        model.scheduler_G.step()
        log(logger, running_dloss_dict, epoch)
        log(logger, running_gloss_dict, epoch)

        # モデルとOptimizerの保存
        if epoch % 10000 == 0:
            model.save_models(epoch, os.path.join(exp_name, 'models'))
            model.save_optims(epoch, os.path.join(exp_name, 'optims'))
