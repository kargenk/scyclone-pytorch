import argparse
import os

import torch
from sklearn.metrics import accuracy_score
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
    Tensor Boardに各損失と訓練状況のログを吐き出す

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

    # コマンドライン引数の受け取り
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', help='experience name', default='MMDD_dataset_batch_option')
    parser.add_argument('--source', help='source directory path', default='data/processed_logmel_kiritan-no7_24k/kiritan/train/')
    parser.add_argument('--target', help='target directory path', default='data/processed_logmel_kiritan-no7_24k/no7/train/')
    parser.add_argument('--device', help='use GPU number', default='0')
    parser.add_argument('--resume_iter', type=int, help='iteration of pretrained model.')
    parser.add_argument('--weights_dir', help='weights directory path')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # データローダーの作成
    dataset = LogMelspDataset(source_dir=args.source,
                              target_dir=args.target)
    print(f'source to target: {len(dataset)}')

    # データローダーの作成
    train_loader = DataLoader(dataset,
                              batch_size=8,  # バッチサイズ
                              shuffle=True,     # データシャッフル
                              num_workers=2,    # 高速化
                              pin_memory=True,  # 高速化
                              worker_init_fn=worker_init_fn)

    print(iter(train_loader).next()[0].shape)
    print(f'iteration size: {len(train_loader)}')

    # モデル，最適化アルゴリズム，スケジューラ，TensorBoardの設定
    model = Scyclone(device=device)
    if args.resume_iter:
        model.restore_all(i=args.resume_iter, checkpoint_dir=args.weights_dir)
    model.configure_optimizers()
    logger = Logger(os.path.join(args.exp_name, 'outputs', 'logs'))

    # 訓練の実行
    is_d_train = True   # Dの訓練フラグ
    is_g_train = False  # Gの訓練フラグ
    epoch = 100000
    for epoch in tqdm(range(1, epoch + 1)):
        logs_D, logs_G = {}, {}
        running_dloss_dict, running_gloss_dict = {}, {}
        for batch in train_loader:
            src, trg = batch
            batch = (src.to(device), trg.to(device))

            # Discriminatorの訓練
            out_d_dict, superior_idx = model.train_d(batch)
            if is_d_train:
                for i in range(model.num_d):
                    model.multi_optim_D_A[i].zero_grad()
                    model.multi_optim_D_B[i].zero_grad()
                out_d_dict['loss'].backward()
                for i in range(model.num_d):
                    if i in superior_idx['A']:
                        model.multi_optim_D_A[i].step()
                        model.multi_scheduler_D_A[i].step()
                    if i in superior_idx['B']:
                        model.multi_optim_D_B[i].step()
                        model.multi_scheduler_D_B[i].step()

            # Generatorの訓練
            out_g_dict = model.train_g(batch)
            if is_g_train:
                model.optim_G.zero_grad()
                out_g_dict['loss'].backward()
                model.optim_G.step()
                model.scheduler_G.step()

            # 各損失ごとにバッチ間で総和を取って集計する(-1: 優秀なDiscriminatorのidリスト)
            running_dloss_dict = loss_dic_update(out_d_dict, running_dloss_dict)
            running_gloss_dict = loss_dic_update(out_g_dict, running_gloss_dict)

        # エポック内での平均値にならす
        for key in running_dloss_dict.keys():
            running_dloss_dict[key] /= len(train_loader)
        for key in running_gloss_dict.keys():
            running_gloss_dict[key] /= len(train_loader)
        logs_D.update(running_dloss_dict)
        logs_G.update(running_gloss_dict)

        # Dの正答率によって学習を切り替え
        with torch.inference_mode():
            out_d_dict, _ = model.train_d(batch)
            preds_real = torch.tensor([]).to(device)
            preds_fake = torch.tensor([]).to(device)
            preds = torch.tensor([]).to(device)
            targets = torch.tensor([]).to(device)
            for key, v in out_d_dict['preds'].items():
                preds = torch.cat([preds, torch.sign(v)], dim=0)
                if 'real' in key:
                    labels = torch.ones_like(v)
                    preds_real = torch.cat([preds_real, torch.sign(v)], dim=0)
                elif 'fake' in key:
                    labels = torch.full_like(v, -1)
                    preds_fake = torch.cat([preds_fake, torch.sign(v)], dim=0)
                else:
                    raise ValueError
                targets = torch.cat([targets, labels], dim=0)
            real_acc = accuracy_score(torch.ones_like(preds_real).cpu().numpy(), preds_real.cpu().numpy())
            fake_acc = accuracy_score(torch.full_like(preds_fake, -1).cpu().numpy(), preds_fake.cpu().numpy())
            acc = accuracy_score(targets.cpu().numpy(), preds.cpu().numpy())

            # D/Gいずれかのモデルがある程度学習した時にはフラグの切り替えとモデルの保存
            if (acc >= 0.8 and is_d_train):
                print('Switching D -> G')
                is_d_train = False
                is_g_train = True
                if epoch >= 300000:
                    model.save_all(epoch, os.path.join(args.exp_name, 'models', 'D_optimised'))
            elif ((real_acc <= 0.5 or fake_acc <= 0.5) and is_g_train):
                print('Switching G -> D')
                is_d_train = True
                is_g_train = False
                if epoch >= 300000:
                    model.save_all(epoch, os.path.join(args.exp_name, 'models', 'G_optimized'))
            else:
                # print('keep learning ...')
                pass

        logs_D['Discriminator/Accuracy'] = acc
        logs_D['Discriminator/FakeAccuracy'] = fake_acc
        logs_D['Discriminator/RealAccuracy'] = real_acc
        logs_D['Discriminator/isLearning'] = is_d_train
        logs_G['Generator/isLearning'] = is_g_train
        for i in range(model.num_d):
            logs_D[f'Discriminator/LearningRate/A_{i}'] = model.multi_scheduler_D_A[i].get_last_lr()[0]
            logs_D[f'Discriminator/LearningRate/B_{i}'] = model.multi_scheduler_D_B[i].get_last_lr()[0]
        logs_G['Generator/LearningRate'] = model.scheduler_G.get_last_lr()[0]

        log(logger, logs_D, epoch)
        log(logger, logs_G, epoch)

        # モデルとOptimizerの保存
        if epoch % 500 == 0:
            model.save_all(epoch, os.path.join(args.exp_name, 'models', 'for_500-epoch'))
