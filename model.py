import itertools
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary


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
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
        x_noised = x + torch.randn(x.size(), device=self.device) * self.noise_sigma
        _x = self.input_layer(x_noised)
        _x = self.res_layer(_x)
        out = self.output_layer(_x)
        return out


class Scyclone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # モデル
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.G_A2B = Generator(num_resblocks=7)
        self.G_B2A = Generator(num_resblocks=7)
        self.D_A = Discriminator(num_resblocks=6)
        self.D_B = Discriminator(num_resblocks=6)

        # 損失の係数
        self.weight_cycle = 10
        self.weight_identity = 1

        # Hinge Lossのオフセット(SNGANは1.0, Scycloneは0.5)
        # ref: https://arxiv.org/abs/2005.03334 eq(2) m
        self.hinge_offset_for_D = 0.5

        # 学習率
        self.learning_rate = 2.0 * 1e-4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.G_A2B(x)

    def save(self, save_path: str) -> None:
        torch.save(self.state_dict(), save_path)

    def load(self, weights_path: str) -> None:
        self.load_state_dict(torch.load(weights_path))

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
        fake_B = self.G_A2B(real_A)
        pred_fake_B = self.D_B(torch.narrow(fake_B, 2, 16, 128))
        loss_adv_G_A2B = torch.mean(F.relu(-1.0 * pred_fake_B))

        fake_A = self.G_B2A(real_B)
        pred_fake_A = self.D_A(torch.narrow(fake_A, 2, 16, 128))
        loss_adv_G_B2A = torch.mean(F.relu(-1.0 * pred_fake_A))

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

        # Total Loss
        loss_G = (
            loss_adv_G_A2B + loss_adv_G_B2A
            + self.weight_cycle * loss_cycle_ABA
            + self.weight_cycle * loss_cycle_BAB
            + self.weight_identity * loss_identity_A
            + self.weight_identity * loss_identity_B
        )

        log = {
            'Loss/G_total': loss_G,
            'Loss/Adv/G_A2B': loss_adv_G_A2B,
            'Loss/Adv/G_B2A': loss_adv_G_B2A,
            'Loss/Cyc/A2B2A': loss_cycle_ABA,
            'Loss/Cyc/B2A2B': loss_cycle_BAB,
            'Loss/Id/A2A': loss_identity_A,
            'Loss/Id/B2B': loss_identity_B,
        }

        out = {'loss': loss_G, 'log': log}

        return out

    def train_d(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> dict:
        """
        Discriminatorの訓練.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): 真の変換元データと真の変換先データのペア

        Returns:
            dict: 損失の合計値と各損失を格納した辞書
        """
        real_A, real_B = batch

        # Adversarial Loss (Hinge Loss)
        m = self.hinge_offset_for_D
        ## D_A
        ### Real Loss
        pred_A_real = self.D_A(torch.narrow(real_A, 2, 16, 128))
        loss_D_A_real = torch.mean(F.relu(m - pred_A_real))
        ### Fake Loss
        fake_A = self.G_B2A(real_B)
        pred_A_fake = self.D_A(torch.narrow(fake_A.detach(), 2, 16, 128))
        loss_D_A_fake = torch.mean(F.relu(m + pred_A_fake))
        ### D_A Total Loss
        loss_D_A = loss_D_A_real + loss_D_A_fake

        ## D_B
        ### Real Loss
        pred_B_real = self.D_B(torch.narrow(real_B, 2, 16, 128))
        loss_D_B_real = torch.mean(F.relu(m - pred_B_real))
        ### Fake Loss
        fake_B = self.G_A2B(real_A)
        pred_B_fake = self.D_B(torch.narrow(fake_B.detach(), 2, 16, 128))
        loss_D_B_fake = torch.mean(F.relu(m + pred_B_fake))
        ### Total Loss
        loss_D_B = loss_D_B_real + loss_D_B_fake

        # Total Loss
        loss_D = loss_D_A + loss_D_B

        log = {
            'Loss/D_total': loss_D,
            'Loss/D_A': loss_D_A,
            'Loss/D_B': loss_D_B,
        }

        out = {'loss': loss_D, 'log': log}

        return out

    def configure_optimizers(self) -> None:
        decay_rate = 0.1
        decay_iter = 100000

        # Generatorの最適化関数とスケジューラ
        optim_G = Adam(
            itertools.chain(self.G_A2B.parameters(), self.G_B2A.parameters()),
            lr=self.learning_rate,
            betas=(0.5, 0.999),
        )
        scheduler_G = {
            'scheduler': StepLR(optim_G, decay_iter, decay_rate),
            'interval': 'step',
        }

        # Discriminatorの最適化関数とスケジューラ
        optim_D = Adam(
            itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
            lr=self.learning_rate,
            betas=(0.5, 0.999),
        )
        scheduler_D = {
            'scheduler': StepLR(optim_D, decay_iter, decay_rate),
            'interval': 'step',
        }

        return [optim_G, optim_D], [scheduler_G, scheduler_D]


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

    # # モデルの概要
    # print('Generator:\n')
    # summary(G, input_size=(160, 80))
    # print('Discriminator:\n')
    # summary(D, input_size=(128, 80))
