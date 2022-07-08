import os

from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LogMelspDataset
from model import Scyclone
from utils import fix_seed, worker_init_fn


if __name__ == '__main__':
    fix_seed(42)
    source_dir = 'data/processed_logmel_jvs_24k/jvs037/'
    target_dir = 'data/processed_logmel_jvs_24k/jvs015/'
    # checkpoint_dir = '0624_jvs037-jvs015_b40_24k_amat_switching_lr2e-4/models/for_50k-epoch/'
    checkpoint_dir = '0628_jvs037-jvs015_b40_24k_amat_lr2e-4/models/for_50k-epoch/'
    g_iter = 200000
    d_iters = range(200000, g_iter + 1, 50000)
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    # データローダーの作成
    dataset = LogMelspDataset(source_dir=source_dir,
                              target_dir=target_dir)
    print(f'source to target: {len(dataset)}')

    # データローダーの作成
    train_loader = DataLoader(dataset,
                              batch_size=1024,  # バッチサイズ
                              shuffle=True,     # データシャッフル
                              num_workers=2,    # 高速化
                              pin_memory=True,  # 高速化
                              worker_init_fn=worker_init_fn)

    print(iter(train_loader).next()[0].shape)
    print(f'iteration size: {len(train_loader)}')

    # モデルの読み込み
    model = Scyclone(device=device)
    model.restore_all(g_iter, checkpoint_dir, map_location=device)

    for d_iter in d_iters:
        print(f'Loading the trained models from step D: {d_iter}, G: {g_iter} ...')
        checkpoint = torch.load(os.path.join(checkpoint_dir, f'{d_iter}_all.pt'), map_location=device)
        for idx in range(model.num_d):
            model.multi_D_A[idx].load_state_dict(checkpoint[f'D_A_{idx}'])
            model.multi_D_B[idx].load_state_dict(checkpoint[f'D_B_{idx}'])
            # model.multi_optim_D_A[idx].load_state_dict(checkpoint[f'optim_D_A_{idx}'])
            # model.multi_optim_D_B[idx].load_state_dict(checkpoint[f'optim_D_B_{idx}'])
            # model.multi_scheduler_D_A[idx].load_state_dict(checkpoint[f'scheduler_D_A_{idx}'])
            # model.multi_scheduler_D_B[idx].load_state_dict(checkpoint[f'scheduler_D_B_{idx}'])
        print('Loaded all models.')

        epoch = 1
        for epoch in tqdm(range(1, epoch + 1)):
            for batch in train_loader:
                src, trg = batch
                batch = (src.to(device), trg.to(device))

            # Dの正答率によって学習を切り替え
            print('-' * 30)
            accs = {}
            with torch.inference_mode():
                out_d_dict, _ = model.train_d(batch)  # pred_A_real, pred_A_fake, pred_B_real, pred_B_fake
                preds = torch.tensor([]).to(device)
                targets = torch.tensor([]).to(device)
                all_targets = torch.tensor([]).to(device)
                all_preds = torch.tensor([]).to(device)
                for key, v in out_d_dict['preds'].items():
                    _preds = torch.cat([preds, torch.sign(v)])
                    all_preds = torch.cat([all_preds, torch.sign(v)])
                    if 'real' in key:
                        labels = torch.ones_like(v)
                    elif 'fake' in key:
                        labels = torch.full_like(v, -1)
                    else:
                        raise ValueError
                    _targets = torch.cat([targets, labels])
                    all_targets = torch.cat([all_targets, labels])
                    _acc = accuracy_score(_targets.cpu().numpy(), _preds.cpu().numpy())
                    accs[key] = _acc
                    print(f'{key} Acc: {_acc}')
                    # v[v < 0] = 0  # Dの各予測値(見やすさのため-1は0とする)
                    print(torch.sign(v))
                    print(v)
                all_acc = accuracy_score(all_targets.cpu().numpy(), all_preds.cpu().numpy())
                print(f'All Acc: {all_acc}')
                # out_g_dict = model.train_g(batch)
                # print(out_g_dict)
            print('-' * 30)
