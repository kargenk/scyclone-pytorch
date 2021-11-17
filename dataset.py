from torch.utils.data import Dataset, DataLoader

from utils import worker_init_fn


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        feature = self.X[index]
        label = self.y[index]
        # 前処理などを書く -----

        # --------------------
        return feature, label


if __name__ == '__main__':
    train_dataset = MyDataset(train_X, train_y)
    test_dataset = MyDataset(test_X, test_y)

    # データローダーの作成
    train_loader = DataLoader(train_dataset,
                              batch_size=16,    # バッチサイズ
                              shuffle=True,     # データシャッフル
                              num_workers=2,    # 高速化
                              pin_memory=True,  # 高速化
                              worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset,
                             batch_size=16,
                             shuffle=False,
                             num_workers=2,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn)
