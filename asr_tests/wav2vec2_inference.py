import torch
import torch.nn.functional as F
import torchaudio


class GreedyCTCDecoder(torch.nn.Module):
    """言語モデルを使用しないデコーダ. 文脈情報は考慮されず貪欲な文字を得る."""
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])


def load_wav(wav_file_path):
    # 音声ファイルの読み込み
    waveform, sample_rate = torchaudio.load(wav_file_path)
    waveform = waveform.to(device)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    return waveform


def allign(x, y):
    import numpy as np
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

    # numpy形式に
    _x = x.squeeze().detach().cpu().numpy()
    _y = y.squeeze().detach().cpu().numpy()

    # 時間的な長さを揃える
    minlen = min(_x.shape[-1], _y.shape[-1])
    _x = _x[:minlen]
    _y = _y[:minlen]
    print(_x.shape)
    print(_y.shape)

    # FastDTWで距離とアライメントパスを計算
    distance, path = fastdtw(_x, _y, dist=euclidean)
    print(path[:100])

    # アライメント
    xs_path = np.array(list(map(lambda p: p[0], path)))
    ys_path = np.array(list(map(lambda p: p[1], path)))
    print(xs_path[:1000])
    print(ys_path[:1000])
    x_alligned = x[:, xs_path]
    y_alligned = y[:, ys_path]

    # アライメント後の音声を保存
    torchaudio.save('./_assets/jvs015_F_test_alligned.wav', x_alligned.cpu(), 24000)
    torchaudio.save('./_assets/jvs037_M_test_alligned.wav', y_alligned.cpu(), 24000)

    return x_alligned, y_alligned


def get_feature(wav_file_path, feature_type, print_str=False):
    waveform = load_wav(wav_file_path)

    # 音素ラベルの確率分布
    with torch.inference_mode():
        emission, _ = model(waveform)  # 出力は各クラスラベルのロジット

    if print_str:
        # デコーダで文字列に
        decoder = GreedyCTCDecoder(labels=bundle.get_labels())
        transcript = decoder(emission[0])
        print(transcript)

    if feature_type == 'last':
        # 中間特徴量の抽出
        with torch.inference_mode():
            features, _ = model.extract_features(waveform)
        last_feature = features[-1]
        print(f'waveform: {len(waveform[0])}')
        print(f'last feature: {last_feature.shape}')
    elif feature_type == 'logits':
        return emission[0]
    elif feature_type == 'probs':
        probs = F.softmax(emission, dim=-1)  # ロジットを確率に
        # 予測されたクラスをプロット
        plot_classification(probs[0][:100, :], wav_file_path.split('/')[-1])
        return probs[0]

    return None


def plot_classification(probs, name):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize
    plt.figure(figsize=(12, 4))
    plt.imshow(probs.cpu().T, norm=Normalize(vmin=0, vmax=1))
    plt.title('Classification result')
    plt.xlabel('Frame (time-axis)')
    plt.ylabel('Class')
    plt.yticks(np.arange(29), bundle.get_labels())
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'{name}_classification_result.png')
    # print('Class labels:', bundle.get_labels())


if __name__ == '__main__':
    SPEECH_FILE = '_assets/jvs037_to_jvs015_at500000_rec_test.wav'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 訓練済みモデルの読み込み
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    print(f'sample rate: {bundle.sample_rate}')
    print(f'labels: {bundle.get_labels()}')
    model = bundle.get_model().to(device)
    # print(model.__class__)

    with torch.inference_mode():
        # # 生成音声(学習初期, ビープ音的なもの)
        # wav_garbage_path = '_assets/jvs037_to_jvs015_at200000_rec_test.wav'
        # feature_garbage = get_feature(wav_garbage_path, 'probs')

        # # 生成音声(学習終期, 人の声には聞こえつつある)
        # wav_rec_path = '_assets/jvs037_to_jvs015_at500000_rec_test.wav'
        # feature_rec = get_feature(wav_rec_path, 'probs')

        # 変換先音声(女性)
        wav_female_path = './_assets/jvs015_F_test_cut.wav'
        feature_female = get_feature(wav_female_path, feature_type='probs', print_str=True)

        # 変換元音声(男性)
        wav_male_path = './_assets/jvs037_M_test_cut.wav'
        feature_male = get_feature(wav_male_path, feature_type='probs', print_str=True)

        # 損失の確認
        print(feature_female.shape)
        print(feature_male.shape)
        # loss_garbage = F.cross_entropy(feature_garbage[:, :100, :], feature_female[:, :100, :])
        # print(f'asr loss (garbage and src): {loss_garbage}')

        # loss_rec = F.cross_entropy(feature_rec[:100, :], feature_female[:100, :])
        # print(f'asr loss (rec and src): {loss_rec}')

        loss_src = F.cross_entropy(feature_female[:200, :], feature_male[:200, :])
        print(f'asr loss (male and female): {loss_src}')

        # # dtwで音声アラインメントを取って保存
        # wav_female = load_wav(wav_female_path)
        # wav_male = load_wav(wav_male_path)
        # wav_female_alligned, wav_male_alligned = allign(wav_female, wav_male)
