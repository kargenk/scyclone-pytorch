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
    # src = 'jvs037'
    # trg = 'jvs015'
    src = 'no7'
    trg = 'pjs'
    base_dir = f'0414_{src}-{trg}_b_24k'
    save_dir = os.path.join(base_dir, 'result')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    resume_iter = 300000
    sr = 24000

    # log melspを抽出して160フレーム毎に分割
    # log_melsp = wav_to_log_melsp(f'../datasets/jvs_music/test_common/{src}/raw.wav', sr=sr)
    log_melsp = wav_to_log_melsp(f'../datasets/kiritan-no7/no7/test/49.wav', sr=sr)
    splited = time_split(log_melsp, 160)

    # [N, 80, 160]に整形
    input_tensor = torch.stack(splited).to(device)

    # 学習済みモデルの読み込み，log melspを変換
    model = Scyclone(device=device)
    model.restore_models(resume_iter, os.path.join(base_dir, 'models'), map_location=device)
    output = model(input_tensor)

    # バッチ次元をつなげて正しい時間軸に整形
    splited = torch.split(output, split_size_or_sections=1, dim=0)
    time_concat = torch.cat(splited, dim=-1)

    # 大きすぎる値をクリッピング
    clipped = clipping_power(time_concat)

    # 音声の保存
    data = wav_from_melsp(clipped)
    sf.write(os.path.join(save_dir, f'{src}_to_{trg}_at{resume_iter}_rec_test.wav'),
             data, sr, subtype='PCM_24')

    # スペクトログラムの保存
    plot_melsp(log_melsp, sr=sr, path=os.path.join(save_dir, 'source_log_melsp'))  # source
    # log_melsp_trg = wav_to_log_melsp(f'../datasets/jvs_music/test_common/{trg}/raw.wav')
    # plot_melsp(log_melsp_trg, sr=24000, path=os.path.join(save_dir, 'target_log_melsp'))  # target
    plot_melsp(clipped, sr=sr, path=os.path.join(save_dir, f'converted_log_melsp_{resume_iter}'))  # converted
