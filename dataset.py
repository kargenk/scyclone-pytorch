import os

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import worker_init_fn


class LogMelspDataset(Dataset):
    def __init__(self, source_dir: str, target_dir: str) -> None:
        super().__init__()
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.src_files = librosa.util.find_files(source_dir, ext='npy')
        self.trg_files = librosa.util.find_files(target_dir, ext='npy')

    def __len__(self):
        return min(len(self.src_files), len(self.trg_files))

    def __getitem__(self, index: int):
        src_data_path = self.src_files[index]  # npy形式ファイルの絶対パスを取得
        trg_data_path = self.trg_files[index]

        # メルスペクトログラム，[batch, 80, 160]
        src_log_melsp = np.load(src_data_path)
        trg_log_melsp = np.load(trg_data_path)
        src_log_melsp = torch.FloatTensor(src_log_melsp)
        trg_log_melsp = torch.FloatTensor(trg_log_melsp)

        return src_log_melsp, trg_log_melsp


if __name__ == '__main__':
    jvs001_to_jvs015_dataset = LogMelspDataset(source_dir='data/processed_logmel/jvs001',
                                     target_dir='data/processed_logmel/jvs015')
    print(f'jvs001 to jvs015: {len(jvs001_to_jvs015_dataset)}')

    # データローダーの作成
    train_loader = DataLoader(jvs001_to_jvs015_dataset,
                              batch_size=16,    # バッチサイズ
                              shuffle=True,     # データシャッフル
                              num_workers=2,    # 高速化
                              pin_memory=True,  # 高速化
                              worker_init_fn=worker_init_fn)

    print(iter(train_loader).next()[0].shape)
    print(iter(train_loader).next()[1].shape)
    print(len(train_loader))
