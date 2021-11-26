import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from dataset import LogMelspDataset
from model import Scyclone
from utils import wav_from_melsp, worker_init_fn, Audio2Mel


def wav_to_log_melsp(wav_path: str, sr: int = 24000) -> torch.Tensor:
    """
    対数メルスペクトログラムを返す.

    Args:
        wav_path (str): wavデータ
        sr (int) : サンプリング周波数

    Returns:
        torch.Tensor: 対数メルスペクトログラム[N, Frequency(80), Time]
    """
    fft = Audio2Mel()

    wav, sr = librosa.load(wav_path, sr=sr, mono=True, dtype=np.float64)
    wav, _ = librosa.effects.trim(wav, top_db=15)  # 無音区間のトリミング(閾値=15dB)
    data_t = torch.from_numpy(wav).float().unsqueeze(0)
    log_melsp = fft(data_t)

    return log_melsp


def time_split(log_melsp: torch.Tensor, frames: int = 160) -> list:
    """
    対数メルスペクトログラムを160フレーム毎のテンソルに分割して返す.

    Args:
        log_melsp (torch.Tensor): 対数メルスペクトログラム[N, Frequency(80), Time]
        frames (int): フレーム数

    Returns:
        list: 160フレーム毎に分割した対数メルスペクトログラムのリスト
    """
    log_melsp = log_melsp.squeeze()
    # 160フレームずつに分ける
    time_splited = []
    for start_idx in range(0, log_melsp.shape[-1] - frames + 1, frames):
        one_audio_seg = log_melsp[:, start_idx: start_idx + frames]

        if one_audio_seg.shape[-1] == frames:
            time_splited.append(one_audio_seg)

    return time_splited


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    resume_iter = 10000

    # log melspを抽出して160フレーム毎に分割
    log_melsp = wav_to_log_melsp('data/train_unique/jvs015/raw.wav')
    splited = time_split(log_melsp, 160)

    # [N, 80, 160]に整形
    input_tensor = torch.stack(splited).to(device)

    # 学習済みモデルの読み込み，log melspを変換
    model = Scyclone()
    model.restore_models(resume_iter, 'models')
    output = model(input_tensor)

    # バッチ次元をつなげて正しい時間軸に整形
    splited = torch.split(output, split_size_or_sections=1, dim=0)
    time_concat = torch.cat(splited, dim=-1)

    # 音声の保存
    data = wav_from_melsp(time_concat)
    sf.write(f'result/jvs015_to_jvs015_at{resume_iter}_rec.wav', data, 24000, subtype='PCM_24')
