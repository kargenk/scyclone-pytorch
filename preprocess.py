import argparse
from datetime import datetime
import glob
import os
import shutil

import librosa
import numpy as np
import torch

from utils import Audio2Mel


def load_wavs(dataset_path: str, sr: int):
    """
        `data`: contains all audios file path.
        `resdict`: contains all wav files.
    """

    data = {}
    with os.scandir(dataset_path) as it:
        for entry in it:
            if entry.is_dir():
                data[entry.name] = []
                with os.scandir(entry.path) as it_f:
                    for onefile in it_f:
                        if onefile.is_file():
                            data[entry.name].append(onefile.path)
    print(f'* Loaded keys: {data.keys()}')
    resdict = {}

    cnt = 0
    for key, value in data.items():
        resdict[key] = {}

        for one_file in value:
            filename = one_file.split('/')[-1].split('.')[0]
            newkey = f'{filename}'
            wav, _ = librosa.load(one_file, sr=sr, mono=True, dtype=np.float64)
            y, _ = librosa.effects.trim(wav, top_db=15)   # 無音区間のトリミング(閾値=15dB)
            # wav = np.append(y[0], y[1:] - 0.97 * y[:-1])  # 直前の0.97掛けを引く

            resdict[key][newkey] = wav
            cnt += 1

    print(f'\n* Total audio files: {cnt}.')
    return resdict


def chunks(iterable, size):
    """
        Yield successive n-sized chunks from iterable.
        sizeサンプルずつ吐き出す．
    """

    for i in range(0, len(iterable), size):
        yield iterable[i: i + size]


def wav_to_melsp_file(dataset: str, sr: int, processed_filepath: str = './data/processed_logmel'):
    """wavから対数メルスペクトログラムに変換する．"""
    CHUNK_SIZE = 1  # 一人当たりの曲数
    FRAMES = 160    # メルスペクトログラムのフレーム数
    fft = Audio2Mel()

    # ファイルの探査
    shutil.rmtree(processed_filepath)
    os.makedirs(processed_filepath, exist_ok=True)

    allwavs_cnt = len(glob.glob(f'{dataset}/*/*.wav'))
    print(f'* Total audio files: {allwavs_cnt}.')

    d = load_wavs(dataset, sr)
    # 1人ずつ処理を行う
    for one_speaker in d.keys():
        values_of_one_speaker = list(d[one_speaker].values())

        # 一曲ずつ(CHUNK_SIZE=1)メルスペクトログラムに変換していく
        for index, one_chunk in enumerate(chunks(values_of_one_speaker, CHUNK_SIZE)):
            wav_concated = []
            temp = one_chunk.copy()

            for one in temp:
                wav_concated.extend(one)
            wav_concated = np.array(wav_concated)
            wav_concated = torch.from_numpy(wav_concated).float().unsqueeze(0)
            print(wav_concated.shape)

            melsp = fft(wav_concated.unsqueeze(0))
            melsp = melsp.squeeze().numpy()
            print('melsp:', melsp.shape, melsp.mean(), melsp.std())

            newname = f'{one_speaker}_{index}'
            file_path_z = os.path.join(processed_filepath, newname)
            np.savez(file_path_z, melsp=melsp)
            print(f'[SAVE]: {file_path_z}')

            # 160フレームずつに分ける
            for start_idx in range(0, melsp.shape[1] - FRAMES + 1, FRAMES):
                one_audio_seg = melsp[:, start_idx: start_idx + FRAMES]

                if one_audio_seg.shape[1] == FRAMES:
                    temp_name = f'{newname}_{start_idx}'
                    file_path = os.path.join(processed_filepath, temp_name)
                    np.save(file_path, one_audio_seg)
                    print(f'[SAVE]: {file_path}.npy')


if __name__ == '__main__':
    input_dir = 'data/train_unique'
    output_dir = 'data/processed_logmel_unique_win1024shift256'
    dataset_default = 'jvs'

    start = datetime.now()

    parser = argparse.ArgumentParser(description='Convert the wav waveform to mel-spectrograms\
    and calculate the speech statistical characteristics.')

    parser.add_argument('--dataset', type=str, default=dataset_default,
                        choices=['VCC2016', 'VCC2018', 'jvs'],
                        help='Available datasets: VCC2016, VCC2018, and jvs (Default: VCC2016).')
    parser.add_argument('--input_dir', type=str,
                        default=input_dir, help='Directory of input data.')
    parser.add_argument('--output_dir', type=str,
                        default=output_dir, help='Directory of processed data.')

    argv = parser.parse_args()
    input_dir = argv.input_dir
    output_dir = argv.output_dir

    os.makedirs(output_dir, exist_ok=True)

    """
        Sample rate:
            VCC2016: 16000 Hz
            VCC2018: 22050 Hz
            jvs:     24000 Hz
    """
    if argv.dataset == 'VCC2016':
        sample_rate = 16000
    elif argv.dataset == 'VCC2018':
        sample_rate = 22050
    elif argv.dataset == 'jvs':
        sample_rate = 24000
    else:
        print('unsupported dataset')
        raise ValueError

    wav_to_melsp_file(input_dir, sample_rate, processed_filepath=output_dir)

    end = datetime.now()
    print(f'* Duration: {end - start}.')