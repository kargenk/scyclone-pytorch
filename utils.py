import random

from librosa.filters import mel as librosa_mel_fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=256,
        hop_length=128,
        win_length=256,
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
    audio = vocoder.inverse(log_melsp).squeeze().cpu()

    return audio


if __name__ == '__main__':
    fix_seed(42)