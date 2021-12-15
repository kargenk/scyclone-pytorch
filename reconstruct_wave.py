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
    # クリッピング前
    plt.figure()
    sns.distplot(melsp.flatten().detach().cpu().numpy())
    plt.savefig('result/power_distribution_before.png')

    # クリッピング後
    plt.figure()
    clipped = torch.clamp(melsp, max=1)
    sns.distplot(clipped.flatten().detach().cpu().numpy())
    plt.savefig('result/power_distribution_after.png')

    return clipped


if __name__ == '__main__':
    if not os.path.exists('result'):
        os.makedirs('result')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    resume_iter = 300000

    # log melspを抽出して160フレーム毎に分割
    log_melsp = wav_to_log_melsp('data/test_common/jvs001/raw.wav')
    # log_melsp = wav_to_log_melsp('data/vcc2018/vcc2018_evaluation/VCC2SM1/30001.wav')
    splited = time_split(log_melsp, 160)

    # [N, 80, 160]に整形
    input_tensor = torch.stack(splited).to(device)

    # 学習済みモデルの読み込み，log melspを変換
    model = Scyclone()
    model.restore_models(resume_iter, '1126_jvs001_to_jvs015/models')
    output = model(input_tensor)

    # バッチ次元をつなげて正しい時間軸に整形
    splited = torch.split(output, split_size_or_sections=1, dim=0)
    time_concat = torch.cat(splited, dim=-1)

    # 大きすぎる値をクリッピング
    clipped = clipping_power(time_concat)

    # 音声の保存
    data = wav_from_melsp(clipped)
    sf.write(f'result/jvs001_to_jvs015_at{resume_iter}_rec.wav', data, 24000, subtype='PCM_24')
    # sf.write(f'result/vcc2sm1_to_vcc2sf1_at{resume_iter}_cripped_rec.wav', data, 24000, subtype='PCM_24')

    # スペクトログラムの保存
    plot_melsp(log_melsp, path='result/source_log_melsp')  # source
    log_melsp_trg = wav_to_log_melsp('data/test_common/jvs015/raw.wav')
    # log_melsp_trg = wav_to_log_melsp('data/vcc2018/vcc2018_evaluation/VCC2SF1/30001.wav')
    plot_melsp(log_melsp_trg, path='result/target_log_melsp')  # target
    plot_melsp(clipped, path=f'result/converted_log_melsp_{resume_iter}')  # converted
