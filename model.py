import torch
import torch.nn as nn
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
    def __init__(self, num_resblocks: int) -> None:
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.input_layer = nn.Sequential(
            nn.Conv1d(160, 256, kernel_size=1, stride=1),
            nn.LeakyReLU(0.01),
        )

        blocks = [Residual() for _ in range(num_resblocks)]
        self.res_layer = nn.Sequential(*blocks)

        self.output_layer = nn.Sequential(
            nn.Conv1d(256, 160, kernel_size=1, stride=1),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = self.input_layer(x)
        _x = self.res_layer(_x)
        out = self.output_layer(_x)
        return out


class Discriminator(nn.Module):
    def __init__(self, num_resblocks: int) -> None:
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.noise_sigma = 0.01

        self.input_layer = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(128, 256, kernel_size=1, stride=1)),
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
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        return x

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load(self, weights_path):
        self.load_state_dict(torch.load(weights_path))


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Generator
    Generator = Generator(num_resblocks=7).to(device)
    print('Generator:\n')
    summary(Generator, input_size=(160, 80))  # 両橋16フレームずつカットしてDiscriminatorに渡す

    # Discriminator
    Discriminator = Discriminator(num_resblocks=6).to(device)
    print('Discriminator:\n')
    summary(Discriminator, input_size=(128, 80))
