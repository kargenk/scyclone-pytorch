import os

import soundfile as sf
import torch

from model import Scyclone
from utils import time_split, wav_from_melsp, wav_to_log_melsp, plot_melsp


if __name__ == '__main__':
    if not os.path.exists('result'):
        os.makedirs('result')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    resume_iter = 300000

    # log melspを抽出して160フレーム毎に分割
    # log_melsp = wav_to_log_melsp('data/train_unique/jvs001/raw.wav')
    log_melsp = wav_to_log_melsp('data/vcc2018/wav_raw/VCC2SM1/10001.wav')
    splited = time_split(log_melsp, 160)

    # [N, 80, 160]に整形
    input_tensor = torch.stack(splited).to(device)

    # 学習済みモデルの読み込み，log melspを変換
    model = Scyclone()
    model.restore_models(resume_iter, '1203_vcc_subset/models')
    output = model(input_tensor)

    # バッチ次元をつなげて正しい時間軸に整形
    splited = torch.split(output, split_size_or_sections=1, dim=0)
    time_concat = torch.cat(splited, dim=-1)

    # 音声の保存
    data = wav_from_melsp(time_concat)
    # sf.write(f'result/jvs001_to_jvs015_at{resume_iter}_rec.wav', data, 24000, subtype='PCM_24')
    sf.write(f'result/vcc2sm1_to_vcc2sf1_at{resume_iter}_rec.wav', data, 24000, subtype='PCM_24')

    # スペクトログラムの保存
    plot_melsp(log_melsp, path='result/source_log_melsp')  # source
    # log_melsp_trg = wav_to_log_melsp('data/train_unique/jvs015/raw.wav')
    log_melsp_trg = wav_to_log_melsp('data/vcc2018/wav_raw/VCC2SF1/10001.wav')
    plot_melsp(log_melsp_trg, path='result/target_log_melsp')  # target
    plot_melsp(time_concat, path=f'result/converted_log_melsp_{resume_iter}')  # converted
