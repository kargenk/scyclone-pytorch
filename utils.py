import random

import librosa
import librosa.display
from librosa.filters import mel as librosa_mel_fn
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,       # scyclone: 256
        hop_length=256,   # scyclone: 128
        win_length=1024,  # scyclone: 256
        sampling_rate=24000,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                             #
        ##############################################
        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))  # db単位に変換
        return log_mel_spec


def fix_seed(seed: int) -> None:
    """
    再現性の担保のために乱数シードの固定する関数.

    Args:
        seed (int): シード値
    """
    random.seed(seed)     # random
    np.random.seed(seed)  # numpy
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """
    データローダーのサブプロセスの乱数seedを固定する関数.

    Args:
        worker_id (int): シード値
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def wav_from_melsp(log_melsp: torch.Tensor) -> torch.Tensor:
    """
    MelGANを用いてlog mel-spectrogramを音声データに変換.

    Args:
        log_melsp (torch.Tensor): shape = [batch_size, 80, timesteps]の対数メルスペクトログラム.

    Returns:
        torch.Tensor: [timesteps]の音声信号
    """
    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    # summary(vocoder.mel2wav, input_size=(80, 10))
    audio = vocoder.inverse(log_melsp).squeeze().cpu().numpy()

    return audio


def wav_to_log_melsp(wav_path: str, sr: int = 24000) -> torch.Tensor:
    """
    対数メルスペクトログラムを返す.

    Args:
        wav_path (str): wavデータ
        sr (int) : サンプリング周波数

    Returns:
        torch.Tensor: 対数メルスペクトログラム[N, Frequency(80), Time]
    """
    fft = Audio2Mel(sampling_rate=sr)

    wav, sr = librosa.load(wav_path, sr=sr, mono=True, dtype=np.float64)
    wav, _ = librosa.effects.trim(wav, top_db=15)  # 音声ファイルの前後の無音区間のトリミング(閾値=15dB)
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


def plot_melsp(log_melsp, sr: int, path: str):
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(log_melsp.cpu().squeeze().detach().numpy(),
                             sr=sr, hop_length=256, x_axis='time', y_axis='linear',
                             norm=Normalize(vmin=-5, vmax=0))
    plt.colorbar(format='%+2.0f dB')
    # plt.xlim(0, 2)
    plt.savefig(f'{path}.png')


if __name__ == '__main__':
    fix_seed(42)

    sr = 24000
    fft = Audio2Mel(sampling_rate=sr)
    wav, sr = librosa.load('../datasets/kiritan-no7/kiritan/test/47.wav', sr=sr, mono=True, dtype=np.float64)
    wav, _ = librosa.effects.trim(wav, top_db=15)  # 音声ファイルの前後の無音区間のトリミング(閾値=15dB)
    sf.write(f'kiritan_24k.wav', wav, 24000, subtype='PCM_24')
    data_t = torch.from_numpy(wav).float().unsqueeze(0)
    log_melsp = fft(data_t)
    wav = wav_from_melsp(log_melsp)
    import soundfile as sf
    sf.write(f'kiritan_24k_rec.wav', wav, 24000, subtype='PCM_24')
