import os

import seaborn as sns
import matplotlib.pyplot as plt
import soundfile as sf
import torch

from model import Scyclone
from utils import time_split, wav_from_melsp, wav_to_log_melsp, plot_melsp


def clipping_power(melsp: torch.Tensor) -> torch.Tensor:
    """
    メルスペクトログラム中の不正な値をクリッピングする

    Args:
        melsp (torch.Tensor): メルスペクトログラム

    Returns:
        torch.Tensor: クリッピング後のメルスペクトログラム
    """
    # # クリッピング前
    # plt.figure()
    # sns.distplot(melsp.flatten().detach().cpu().numpy())
    # plt.savefig('result/power_distribution_before.png')

    # # クリッピング後
    # plt.figure()
    clipped = torch.clamp(melsp, max=1)
    # sns.distplot(clipped.flatten().detach().cpu().numpy())
    # plt.savefig('result/power_distribution_after.png')

    return clipped


if __name__ == '__main__':
    src = 'jvs037'
    trg = 'jvs015'
    flag = 'test'
    base_dir = f'0706_{src}-{trg}_b40_24k_amat_switching_lr2e-4'
    checkpoint_dir = os.path.join(base_dir, 'models', 'for_50k-epoch')
    save_dir = os.path.join(base_dir, 'result', flag)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    resume_iter = 100000
    sr = 24000

    # log melspを抽出して160フレーム毎に分割
    if flag == 'train':
        log_melsp = wav_to_log_melsp(f'../datasets/jvs_music/train_unique/{src}/raw.wav', sr=sr)
        log_melsp_trg = wav_to_log_melsp(f'../datasets/jvs_music/train_unique/{trg}/raw.wav', sr=sr)
    elif flag == 'test':
        log_melsp = wav_to_log_melsp(f'../datasets/jvs_music/test_common/{src}/raw.wav', sr=sr)
        log_melsp_trg = wav_to_log_melsp(f'../datasets/jvs_music/test_common/{trg}/raw.wav', sr=sr)
    else:
        ValueError
    splited = time_split(log_melsp, 160)

    # [N, 80, 160]に整形
    input_tensor = torch.stack(splited).to(device)

    # 学習済みモデルの読み込み，log melspを変換
    model = Scyclone(device=device)
    model.restore_all(resume_iter, checkpoint_dir, map_location=device)
    output = model(input_tensor)

    # バッチ次元をつなげて正しい時間軸に整形
    splited = torch.split(output, split_size_or_sections=1, dim=0)
    time_concat = torch.cat(splited, dim=-1)

    # 大きすぎる値をクリッピング
    clipped = clipping_power(time_concat)

    # 音声の保存
    data = wav_from_melsp(clipped)
    sf.write(os.path.join(save_dir, f'{src}_to_{trg}_at{resume_iter}_rec_{flag}.wav'),
             data, sr, subtype='PCM_24')

    # スペクトログラムの保存
    plot_melsp(log_melsp, sr=sr, path=os.path.join(save_dir, f'source_log_melsp_{flag}'))  # source
    plot_melsp(log_melsp_trg, sr=24000, path=os.path.join(save_dir, f'target_log_melsp_{flag}'))  # target
    plot_melsp(clipped, sr=sr, path=os.path.join(save_dir, f'converted_log_melsp_{resume_iter}_{flag}'))  # converted
