import itertools
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary

from utils import wav_from_melsp


class Residual(nn.Module):
    """ Generator用のResBlock. """
    def __init__(self) -> None:
        super().__init__()

        self.resblock = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=5//2),
            nn.LeakyReLU(0.01),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=5//2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.resblock(x)


class ResidualSN(nn.Module):
    """ Discriminator用のResBlock. """
    def __init__(self) -> None:
        super().__init__()

        self.resblock = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=5//2)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=5//2)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.resblock(x)


class Generator(nn.Module):
    def __init__(self, num_resblocks: int = 7) -> None:
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv1d(80, 256, kernel_size=1, stride=1),
            nn.LeakyReLU(0.01),
        )

        blocks = [Residual() for _ in range(num_resblocks)]
        self.res_layer = nn.Sequential(*blocks)

        self.output_layer = nn.Sequential(
            nn.Conv1d(256, 80, kernel_size=1, stride=1),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = self.input_layer(x)
        _x = self.res_layer(_x)
        out = self.output_layer(_x)
        return out


class Discriminator(nn.Module):
    def __init__(self, num_resblocks: int = 6) -> None:
        super().__init__()
        self.noise_sigma = 0.01

        self.input_layer = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(80, 256, kernel_size=1, stride=1)),
            nn.LeakyReLU(0.2),
        )

        blocks = [ResidualSN() for _ in range(num_resblocks)]
        self.res_layer = nn.Sequential(*blocks)

        self.output_layer = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(256, 1, kernel_size=1, stride=1)),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(output_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 不安定さを解決するために入力にN(0, 0.01)のガウシアンノイズを加える(論文より)
        x_noised = x + torch.randn(x.size(), device=x.device) * self.noise_sigma
        _x = self.input_layer(x_noised)
        _x = self.res_layer(_x)
        out = self.output_layer(_x)
        return out


class Scyclone(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device
        # モデル
        self.num_d = 3
        self.G_A2B = Generator(num_resblocks=7).to(self.device)
        self.G_B2A = Generator(num_resblocks=7).to(self.device)
        self.multi_D_A, self.multi_D_B = [], []
        for _ in range(self.num_d):
            self.multi_D_A.append(Discriminator(num_resblocks=6).to(self.device))
            self.multi_D_B.append(Discriminator(num_resblocks=6).to(self.device))

        # 補助損失用の音声認識モデル
        self.asr_model = self.get_asr_model()

        # メルスペクトログラムからの音声復元用のモデル
        self.vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

        # 損失の係数
        self.weight_cycle = 10
        self.weight_identity = 1

        # Hinge Lossのオフセット(SNGANは1.0, Scycloneは0.5)
        # ref: https://arxiv.org/abs/2005.03334 eq(2) m
        self.hinge_offset_for_D = 0.5

        # 学習率
        self.learning_rate = 2.0 * 1e-4  # default: 2.0 * 1e-4

        self.configure_optimizers()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.G_A2B(x)

    def save_all(self, i: int, save_dir: str) -> None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        checkpoint = {
            'D_A_0': self.multi_D_A[0].state_dict(),
            'D_A_1': self.multi_D_A[1].state_dict(),
            'D_A_2': self.multi_D_A[2].state_dict(),
            'D_B_0': self.multi_D_B[0].state_dict(),
            'D_B_1': self.multi_D_B[1].state_dict(),
            'D_B_2': self.multi_D_B[2].state_dict(),
            'G_A2B': self.G_A2B.state_dict(),
            'G_B2A': self.G_B2A.state_dict(),
            'optim_D_A_0': self.multi_optim_D_A[0].state_dict(),
            'optim_D_A_1': self.multi_optim_D_A[1].state_dict(),
            'optim_D_A_2': self.multi_optim_D_A[2].state_dict(),
            'optim_D_B_0': self.multi_optim_D_B[0].state_dict(),
            'optim_D_B_1': self.multi_optim_D_B[1].state_dict(),
            'optim_D_B_2': self.multi_optim_D_B[2].state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'scheduler_D_A_0': self.multi_scheduler_D_A[0].state_dict(),
            'scheduler_D_A_1': self.multi_scheduler_D_A[1].state_dict(),
            'scheduler_D_A_2': self.multi_scheduler_D_A[2].state_dict(),
            'scheduler_D_B_0': self.multi_scheduler_D_B[0].state_dict(),
            'scheduler_D_B_1': self.multi_scheduler_D_B[1].state_dict(),
            'scheduler_D_B_2': self.multi_scheduler_D_B[2].state_dict(),
            'scheduler_G': self.scheduler_G.state_dict(),
        }
        torch.save(checkpoint, os.path.join(save_dir, f'{i}_all.pt'))
        print(f'Save model checkpoints into {save_dir}...')

    def restore_all(self, i: int, checkpoint_dir: str, map_location = torch.device('cpu')) -> None:
        print(f'Loading the all conditions on epoch {i} ...')
        checkpoint = torch.load(os.path.join(checkpoint_dir, f'{i}_all.pt'), map_location=map_location)
        self.G_A2B.load_state_dict(checkpoint['G_A2B'])
        self.G_B2A.load_state_dict(checkpoint['G_B2A'])
        self.optim_G.load_state_dict(checkpoint['optim_G'])
        self.scheduler_G.load_state_dict(checkpoint['scheduler_G'])
        for idx in range(self.num_d):
            self.multi_D_A[idx].load_state_dict(checkpoint[f'D_A_{idx}'])
            self.multi_D_B[idx].load_state_dict(checkpoint[f'D_B_{idx}'])
            self.multi_optim_D_A[idx].load_state_dict(checkpoint[f'optim_D_A_{idx}'])
            self.multi_optim_D_B[idx].load_state_dict(checkpoint[f'optim_D_B_{idx}'])
            self.multi_scheduler_D_A[idx].load_state_dict(checkpoint[f'scheduler_D_A_{idx}'])
            self.multi_scheduler_D_B[idx].load_state_dict(checkpoint[f'scheduler_D_B_{idx}'])
        print(f'Successfully Loading.')

    def train_g(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> dict:
        """
        Generatorの訓練.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): 真の変換元データと真の変換先データのペア

        Returns:
            dict: 損失の合計値と各損失を格納した辞書
        """
        real_A, real_B = batch

        # Adversarial Loss (Hinge Loss)
        pred_A_fakes, pred_B_fakes = [], []
        fake_A = self.G_B2A(real_B)
        fake_B = self.G_A2B(real_A)
        lit_A = np.zeros(self.num_d)
        lit_B = np.zeros(self.num_d)
        for i in range(self.num_d):
            # Dの勾配情報は保持しないようにする
            for p_A, p_B in zip(self.multi_D_A[i].parameters(), self.multi_D_B[i].parameters()):
                p_A.requires_grad = False
                p_B.requires_grad = False
            pred_A_fake = self.multi_D_A[i](torch.narrow(fake_A, 2, 16, 128))
            pred_A_fakes.append(pred_A_fake)
            lit_A[i] = torch.sum(pred_A_fake).item()
            pred_B_fake = self.multi_D_B[i](torch.narrow(fake_B, 2, 16, 128))
            pred_B_fakes.append(pred_B_fake)
            lit_B[i] = torch.sum(pred_B_fake).item()

        # 再重み付け
        loss_sort_A = np.argsort(lit_A)
        weights_A = np.random.dirichlet(np.ones(self.num_d))
        weights_A = np.sort(weights_A)[::-1]
        loss_sort_B = np.argsort(lit_B)
        weights_B = np.random.dirichlet(np.ones(self.num_d))
        weights_B = np.sort(weights_B)[::-1]

        is_empty_A, is_empty_B = True, True
        # D_A
        for i in range(len(pred_A_fakes)):
            if is_empty_A == True:
                critic_fake_A = weights_A[i] * pred_A_fakes[loss_sort_A[i]]
                is_empty_A = False
            else:
                critic_fake_A = torch.add(critic_fake_A, weights_A[i] * pred_A_fakes[loss_sort_A[i]])
        loss_adv_G_B2A = torch.mean(F.relu(-1.0 * critic_fake_A))

        # D_B
        for i in range(len(pred_B_fakes)):
            if is_empty_B == True:
                critic_fake_B = weights_B[i] * pred_B_fakes[loss_sort_B[i]]
                is_empty_B = False
            else:
                critic_fake_B = torch.add(critic_fake_B, weights_B[i] * pred_B_fakes[loss_sort_B[i]])
        loss_adv_G_A2B = torch.mean(F.relu(-1.0 * critic_fake_B))

        # Cycle Consistency Loss (L1 Loss)
        cycled_A = self.G_B2A(fake_B)
        loss_cycle_ABA = F.l1_loss(cycled_A, real_A)

        cycled_B = self.G_A2B(fake_A)
        loss_cycle_BAB = F.l1_loss(cycled_B, real_B)

        # Identity Loss (L1 Loss)
        identity_B = self.G_A2B(real_B)
        loss_identity_B = F.l1_loss(identity_B, real_B)

        identity_A = self.G_B2A(real_A)
        loss_identity_A = F.l1_loss(identity_A, real_A)

        # ASR Loss (L1 Loss)
        # 音声認識モデルを用いてその出力が同じ特徴になるように誘導して発話の明瞭性向上を図る
        waveform_fake_A = torch.from_numpy(self.mel2wav(self.vocoder, fake_A))
        waveform_fake_B = torch.from_numpy(self.mel2wav(self.vocoder, fake_B))
        waveform_real_A = torch.from_numpy(self.mel2wav(self.vocoder, real_A))
        waveform_real_B = torch.from_numpy(self.mel2wav(self.vocoder, real_B))
        # まとめて計算できるようにバッチ方向で結合, [N, Time] -> [N * 4, Time]
        waveforms = torch.cat([waveform_fake_A, waveform_fake_B,
                               waveform_real_A, waveform_real_B], dim=0)
        # 中間特徴量の抽出
        with torch.inference_mode():
            features, _ = self.asr_model.extract_features(waveforms.to(self.device))
        last_feature = features[-1]  # 最終層の特徴だけ取り出し
        fake_features, real_features = torch.split(last_feature, real_A.shape[0] * 2, dim=0)  # [N * 4, Time] -> 2 * [N * 2, Time]
        loss_asr = F.l1_loss(fake_features.float(), real_features.float())

        # Total Loss
        loss_G = (
            loss_adv_G_A2B + loss_adv_G_B2A
            + self.weight_cycle * loss_cycle_ABA
            + self.weight_cycle * loss_cycle_BAB
            + self.weight_identity * loss_identity_A
            + self.weight_identity * loss_identity_B
            + loss_asr
        )

        log = {
            'Generator/Loss/G_total': loss_G,
            'Generator/Loss/Adv/G_A2B': loss_adv_G_A2B,
            'Generator/Loss/Adv/G_B2A': loss_adv_G_B2A,
            'Generator/Loss/Cyc/A2B2A': loss_cycle_ABA,
            'Generator/Loss/Cyc/B2A2B': loss_cycle_BAB,
            'Generator/Loss/Id/A2A': loss_identity_A,
            'Generator/Loss/Id/B2B': loss_identity_B,
            'Generator/Loss/ASR': loss_asr,
        }

        out = {'loss': loss_G, 'log': log}

        return out

    def train_d(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> dict:
        """
        Discriminatorの訓練.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): 真の変換元データと真の変換先データのペア

        Returns:
            dict: 損失の合計値と各損失, 全ての予測結果を格納した辞書
            dict: 各クラス(A, B)における優秀なDiscriminatorのidを格納した辞書
        """
        for i in range(self.num_d):
            self.multi_D_A[i].train()
            self.multi_D_B[i].train()
            for p_A, p_B in zip(self.multi_D_A[i].parameters(), self.multi_D_B[i].parameters()):
                p_A.requires_grad = True
                p_B.requires_grad = True

        real_A, real_B = batch

        # Adversarial Loss (Hinge Loss)
        m = self.hinge_offset_for_D

        pred_A_fake, pred_A_real = None, None
        pred_B_fake, pred_B_real = None, None
        fake_A = self.G_B2A(real_B)
        fake_B = self.G_A2B(real_A)

        # 各Discriminatorからの出力をひとまとめにする[batch, num_d]
        is_empty_A, is_empty_B = True, True
        for i in range(self.num_d):
            # D_A
            if is_empty_A:
                pred_A_fake = self.multi_D_A[i](torch.narrow(fake_A.detach(), 2, 16, 128)).unsqueeze(1)
                pred_A_real = self.multi_D_A[i](torch.narrow(real_A, 2, 16, 128)).unsqueeze(1)
                is_empty_A = False
            else:
                pred_A_fake = torch.cat((pred_A_fake, self.multi_D_A[i](torch.narrow(fake_A.detach(), 2, 16, 128)).unsqueeze(1)), axis=1)
                pred_A_real = torch.cat((pred_A_real, self.multi_D_A[i](torch.narrow(real_A, 2, 16, 128)).unsqueeze(1)), axis=1)
            # D_B
            if is_empty_B:
                pred_B_fake = self.multi_D_B[i](torch.narrow(fake_B.detach(), 2, 16, 128)).unsqueeze(1)
                pred_B_real = self.multi_D_B[i](torch.narrow(real_B, 2, 16, 128)).unsqueeze(1)
                is_empty_B = False
            else:
                pred_B_fake = torch.cat((pred_B_fake, self.multi_D_B[i](torch.narrow(fake_B.detach(), 2, 16, 128)).unsqueeze(1)), axis=1)
                pred_B_real = torch.cat((pred_B_real, self.multi_D_B[i](torch.narrow(real_B, 2, 16, 128)).unsqueeze(1)), axis=1)

        # 出力スコアが最も低い(fakeをfakeと正しく判定している)Discriminatorを選択
        superior_A_idx = torch.argmin(pred_A_fake, dim=1)
        mask_A = torch.zeros((real_A.size()[0], self.num_d)).to(self.device)
        superior_B_idx = torch.argmin(pred_B_fake, dim=1)
        mask_B = torch.zeros((real_B.size()[0], self.num_d)).to(self.device)

        # 1バッチ毎にε:0.3で優秀なD, 1-εでランダムなD_Aを選択
        for i in range(mask_A.size()[0]):
            random_checker = np.random.randint(0, 10)
            if random_checker > 7:
                random_idx = np.random.randint(0, self.num_d)
                mask_A[i][random_idx] = 1.0
            else:
                mask_A[i][superior_A_idx[i]] = 1.0

        # 1バッチ毎にε:0.3で優秀なD, 1-εでランダムなD_Bを選択
        for i in range(mask_B.size()[0]):
            random_checker = np.random.randint(0, 10)
            if random_checker > 7:
                random_idx = np.random.randint(0, self.num_d)
                mask_B[i][random_idx] = 1.0
            else:
                mask_B[i][superior_B_idx[i]] = 1.0

        pred_A_fake = pred_A_fake.squeeze()
        pred_A_real = pred_A_real.squeeze()
        pred_A_fake_output = torch.sum(mask_A * pred_A_fake, dim=1)
        pred_A_real_output = torch.sum(mask_A * pred_A_real, dim=1)

        pred_B_fake = pred_B_fake.squeeze()
        pred_B_real = pred_B_real.squeeze()
        pred_B_fake_output = torch.sum(mask_B * pred_B_fake, dim=1)
        pred_B_real_output = torch.sum(mask_B * pred_B_real, dim=1)

        # D_A
        loss_D_A_real = torch.mean(F.relu(m - pred_A_real_output))  # real loss
        loss_D_A_fake = torch.mean(F.relu(m + pred_A_fake_output))  # fake loss
        loss_D_A = loss_D_A_real + loss_D_A_fake  # D_A total

        # D_B
        loss_D_B_real = torch.mean(F.relu(m - pred_B_real_output))  # real loss
        loss_D_B_fake = torch.mean(F.relu(m + pred_B_fake_output))  # fake loss
        loss_D_B = loss_D_B_real + loss_D_B_fake  # D_B total

        # Total Loss
        loss_D = loss_D_A + loss_D_B

        log = {
            'Discriminator/Loss/D_selected_total': loss_D,
            'Discriminator/Loss/D_A_selected': loss_D_A,
            'Discriminator/Loss/D_B_selected': loss_D_B,
        }

        # preds = {'pred_A_real': torch.sign(pred_A_real_output.squeeze()),
        #          'pred_A_fake': torch.sign(pred_A_fake_output.squeeze()),
        #          'pred_B_real': torch.sign(pred_B_real_output.squeeze()),
        #          'pred_B_fake': torch.sign(pred_B_fake_output.squeeze())}

        preds = {'pred_A_real': pred_A_real_output.squeeze(),
                 'pred_A_fake': pred_A_fake_output.squeeze(),
                 'pred_B_real': pred_B_real_output.squeeze(),
                 'pred_B_fake': pred_B_fake_output.squeeze()}

        superior_idx = {
            'A': superior_A_idx,
            'B': superior_B_idx,
        }

        out = {'loss': loss_D, 'log': log, 'preds': preds}, superior_idx

        return out

    def configure_optimizers(self) -> None:
        decay_rate = 0.5
        decay_epoch = 100000

        # Generatorの最適化関数とスケジューラ
        self.optim_G = Adam(
            itertools.chain(self.G_A2B.parameters(), self.G_B2A.parameters()),
            lr=self.learning_rate,
            betas=(0.5, 0.999),
        )
        self.scheduler_G = StepLR(self.optim_G, decay_epoch, decay_rate)

        # Discriminatorの最適化関数とスケジューラ
        self.multi_optim_D_A, self.multi_scheduler_D_A = [], []
        self.multi_optim_D_B, self.multi_scheduler_D_B = [], []
        for idx in range(self.num_d):
            self.multi_optim_D_A.append(
                Adam(
                    self.multi_D_A[idx].parameters(),
                    lr=self.learning_rate,
                    betas=(0.5, 0.999),)
                )
            self.multi_optim_D_B.append(
                Adam(
                    self.multi_D_B[idx].parameters(),
                    lr=self.learning_rate,
                    betas=(0.5, 0.999),)
                )
            self.multi_scheduler_D_A.append(StepLR(self.multi_optim_D_A[idx], decay_epoch, decay_rate))
            self.multi_scheduler_D_B.append(StepLR(self.multi_optim_D_B[idx], decay_epoch, decay_rate))

    def get_asr_model(self):
        model = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
        model = model.to(self.device)

        # 学習中に更新しないように重みを固定
        for param in model.parameters():
            param.requires_grad = False

        return model


    def mel2wav(self, vocoder, log_melsp):
        audio = vocoder.inverse(log_melsp).squeeze().cpu().numpy()
        return audio


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Generator
    dummy_input = torch.randn(1, 80, 160).to(device)
    G = Generator(num_resblocks=7).to(device)
    g_output = G(dummy_input)
    print(f'G input:  {dummy_input.shape}')
    print(f'G output: {g_output.shape}')

    # Discriminator
    D = Discriminator(num_resblocks=6).to(device)
    d_input = torch.narrow(g_output, 2, 16, 128)  # paddingの影響を回避するため，両橋16フレームずつカットしてDiscriminatorに渡す
    d_output = D(d_input)
    print(f'D input:  {d_input.shape}')
    print(f'D output: {d_output.shape}')

    # Scyclone
    S = Scyclone(device=device)
    summary(S, input_size=(80, 160))
    S.save_all(765, 'test')
    S.restore_all(765, 'test')
    # S.restore_models(0, 'models')

    # # モデルの概要
    # print('Generator:\n')
    # summary(G, input_size=(80, 160))
    # print('Discriminator:\n')
    # summary(D, input_size=(80, 128))
