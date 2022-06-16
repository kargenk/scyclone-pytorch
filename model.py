import itertools
import os
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
        self.G_A2B = Generator(num_resblocks=7).to(self.device)
        self.G_B2A = Generator(num_resblocks=7).to(self.device)
        self.D_A = Discriminator(num_resblocks=6).to(self.device)
        self.D_B = Discriminator(num_resblocks=6).to(self.device)

        # 損失の係数
        self.weight_cycle = 10
        self.weight_identity = 1

        # Hinge Lossのオフセット(SNGANは1.0, Scycloneは0.5)
        # ref: https://arxiv.org/abs/2005.03334 eq(2) m
        self.hinge_offset_for_D = 0.5

        # 学習率
        self.learning_rate = 2.0 * 1e-4  # default: 2.0 * 1e-4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.G_A2B(x)

    def save_all(self, i: int, save_dir: str) -> None:
        checkpoint = {
            'D_A': self.D_A.state_dict(),
            'D_B': self.D_B.state_dict(),
            'G_A2B': self.G_A2B.state_dict(),
            'G_B2A': self.G_B2A.state_dict(),
            'optimi_D': self.optim_D.state_dict(),
            'optimi_G': self.optim_G.state_dict(),
            'scheduler_D': self.scheduler_D.state_dict(),
            'scheduler_G': self.scheduler_G.state_dict(),
        }
        torch.save(checkpoint, os.path.join(save_dir, f'{i}_all.pt'))
        print(f'Save model checkpoints into {save_dir}...')

    def save_models(self, i: int, save_dir: str) -> None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        G_A2B_path = os.path.join(save_dir, f'{i}-G_A2B.pt')
        G_B2A_path = os.path.join(save_dir, f'{i}-G_B2A.pt')
        D_A_path = os.path.join(save_dir, f'{i}-D_A.pt')
        D_B_path = os.path.join(save_dir, f'{i}-D_B.pt')
        torch.save(self.G_A2B.state_dict(), G_A2B_path)
        torch.save(self.G_B2A.state_dict(), G_B2A_path)
        torch.save(self.D_A.state_dict(), D_A_path)
        torch.save(self.D_B.state_dict(), D_B_path)
        print(f'Save model checkpoints into {save_dir}...')

    def save_optims(self, i: int, save_dir: str) -> None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        optim_G_path = os.path.join(save_dir, f'{i}-optim_G.pt')
        optim_D_path = os.path.join(save_dir, f'{i}-optim_D.pt')
        torch.save(self.optim_G.state_dict(), optim_G_path)
        torch.save(self.optim_D.state_dict(), optim_D_path)
        print(f'Save optimizer checkpoints into {save_dir}...')

    def restore_models(self, i: int, weights_dir: str, map_location = torch.device('cpu')) -> None:
        print(f'Loading the trained models from step {i} ...')
        G_A2B_path = os.path.join(weights_dir, f'{i}-G_A2B.pt')
        G_B2A_path = os.path.join(weights_dir, f'{i}-G_B2A.pt')
        D_A_path = os.path.join(weights_dir, f'{i}-D_A.pt')
        D_B_path = os.path.join(weights_dir, f'{i}-D_B.pt')
        self.G_A2B.load_state_dict(torch.load(G_A2B_path, map_location=map_location))
        self.G_B2A.load_state_dict(torch.load(G_B2A_path, map_location=map_location))
        self.D_A.load_state_dict(torch.load(D_A_path, map_location=map_location))
        self.D_B.load_state_dict(torch.load(D_B_path, map_location=map_location))
        print('Loaded all models.')

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
            'Generator/Loss/G_total': loss_G,
            'Generator/Loss/Adv/G_A2B': loss_adv_G_A2B,
            'Generator/Loss/Adv/G_B2A': loss_adv_G_B2A,
            'Generator/Loss/Cyc/A2B2A': loss_cycle_ABA,
            'Generator/Loss/Cyc/B2A2B': loss_cycle_BAB,
            'Generator/Loss/Id/A2A': loss_identity_A,
            'Generator/Loss/Id/B2B': loss_identity_B,
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
        ### D_B Total Loss
        loss_D_B = loss_D_B_real + loss_D_B_fake

        # Total Loss
        loss_D = loss_D_A + loss_D_B

        log = {
            'Discriminator/Loss/D_total': loss_D,
            'Discriminator/Loss/D_A': loss_D_A,
            'Discriminator/Loss/D_B': loss_D_B,
        }

        preds = {'pred_A_real': torch.sign(pred_A_real.squeeze()),
                 'pred_A_fake': torch.sign(pred_A_fake.squeeze()),
                 'pred_B_real': torch.sign(pred_B_real.squeeze()),
                 'pred_B_fake': torch.sign(pred_B_fake.squeeze())}

        # preds = {'pred_A_real': pred_A_real.squeeze(),
        #          'pred_A_fake': pred_A_fake.squeeze(),
        #          'pred_B_real': pred_B_real.squeeze(),
        #          'pred_B_fake': pred_B_fake.squeeze()}

        out = {'loss': loss_D, 'log': log, 'preds': preds}

        return out

    def configure_optimizers(self) -> None:
        decay_rate = 0.5
        decay_epoch = 50000

        # Generatorの最適化関数とスケジューラ
        self.optim_G = Adam(
            itertools.chain(self.G_A2B.parameters(), self.G_B2A.parameters()),
            lr=self.learning_rate,
            betas=(0.5, 0.999),
        )
        self.scheduler_G = StepLR(self.optim_G, decay_epoch, decay_rate)

        # Discriminatorの最適化関数とスケジューラ
        self.optim_D = Adam(
            itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
            lr=self.learning_rate,
            betas=(0.5, 0.999),
        )
        self.scheduler_D = StepLR(self.optim_D, decay_epoch, decay_rate)


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
    # S.save_models(0, 'models')
    # S.restore_models(0, 'models')

    # # モデルの概要
    # print('Generator:\n')
    # summary(G, input_size=(80, 160))
    # print('Discriminator:\n')
    # summary(D, input_size=(80, 128))
