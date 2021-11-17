import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LogMelspDataset
from model import Scyclone
from utils import worker_init_fn


def train_model(model, train_loader, test_loader, device):
    # Train loop ----------------------------
    model.train()
    train_batch_loss = 0
    for data, label in train_loader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_batch_loss.append(loss.item())

    # Test(val) loop ----------------------------
    model.eval()
    test_batch_loss = 0
    with torch.inference_mode():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label).item()
            test_batch_loss.append(loss.item())

    return model, np.mean(train_batch_loss), np.mean(test_batch_loss)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # データローダーの作成
    jvs001_to_jvs015_dataset = LogMelspDataset(source_dir='data/processed_logmel/jvs001',
                                     target_dir='data/processed_logmel/jvs015')
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

    # モデル，最適化アルゴリズム，スケジューラの設定
    model = Scyclone()
    optims, schedulers = model.configure_optimizers()
    optim_G, optim_D = optims
    scheduler_G, scheduler_D = schedulers

    # 訓練の実行
    epoch = 1
    for epoch in tqdm(range(epoch)):
        for batch in train_loader:
            # TODO: optimizerの初期化
            # TODO: 各optimizerを進める
            # TODO: 各schedulerを進める
            # TODO: 各損失をTensorBoaedに書き出し
            src, trg = batch
            batch = (src.to(device), trg.to(device))
            out_d_dict = model.train_d(batch)
            out_g_dict = model.train_g(batch)
            print(out_d_dict)
            print(out_g_dict)
