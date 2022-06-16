import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


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
    # torch.backends.cudnn.benchmark = False


class toy_dataset(Dataset):
    def __init__(self, points):
        self.points = points
        print(f'data size: {len(points)}')

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx]


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(args.Z_dim, args.dim1),
            #  nn.BatchNorm1d(args.dim1, 0.8),
            nn.LeakyReLU(),
            nn.Linear(args.dim1, args.dim2),
            #  nn.BatchNorm1d(args.dim2, 0.8),
            nn.LeakyReLU(),
            nn.Linear(args.dim2, args.dim3),
            #  nn.BatchNorm1d(args.dim3, 0.8),
            nn.LeakyReLU(),
            nn.Linear(args.dim3, args.dim3),
            nn.LeakyReLU(),
            nn.Linear(args.dim3, args.dim3),
            nn.LeakyReLU(),
            nn.Linear(args.dim3, args.dim3),
            nn.LeakyReLU(),
            nn.Linear(args.dim3, args.out_dim),
            #  nn.Sigmoid,
            nn.Tanh(),
        )

    def forward(self, z):

        img_gen = self.model(z)
        return 2 * img_gen


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(args.img_dim, args.dim1),
            #  nn.BatchNorm1d(args.dim1, 0.8),
            nn.ReLU(),
            nn.Linear(args.dim1, args.dim2),
            #  nn.BatchNorm1d(args.dim2, 0.8),
            nn.ReLU(),
            nn.Linear(args.dim2, args.dim3),
            nn.ReLU(),
            nn.Linear(args.dim3, args.dim3),
            nn.ReLU(),
            nn.Linear(args.dim3, args.dim3),
            nn.ReLU(),
            nn.Linear(args.dim3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.model(x)
        return out


class Flow(nn.Module):
    def __init__(self):
        super(Flow, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        img_gen = self.model(z)
        return img_gen


def get_parser():
    # コマンドライン引数受け取り
    parser = argparse.ArgumentParser(description="CNN text classificer")

    # 学習関連
    parser.add_argument("--device", type=int, default=0, help="number of GPU device [default: 0]",)
    parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate [default: 0.0001]")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs for train [default: 300]")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for training [default: 128]")
    parser.add_argument("--Z_dim", type=int, default=10)
    parser.add_argument("--dim1", type=int, default=128)
    parser.add_argument("--dim2", type=int, default=512)
    parser.add_argument("--dim3", type=int, default=1024)
    parser.add_argument("--out_dim", type=int, default=2)
    parser.add_argument("--img_dim", type=int, default=2)

    # # データ関連
    # parser.add_argument(
    #     "-shuffle", action="store_true", default=True, help="shuffle the data every epoch"
    # )
    # # モデル関連
    # parser.add_argument(
    #     "-dropout",
    #     type=float,
    #     default=0.5,
    #     help="the probability for dropout [default: 0.5]",
    # )

    return parser


def gaussians(batch_size=1024, flag=True):
    """ガウシアンリングを作成する関数."""
    scale = 2.0
    centers1 = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]

    centers2 = []
    theta = np.pi / 8
    a = np.cos(theta)
    b = np.sin(theta)
    for p in centers1:
        centers2.append((p[0] * a - p[1] * b, p[1] * a + p[0] * b))

    centers1 = [(scale * x, scale * y) for x, y in centers1]
    centers2 = [(scale * x, scale * y) for x, y in centers2]
    count = 0
    while True:
        count += 1
        dataset = []
        for i in range(batch_size):
            point = np.random.randn(2) * 0.05
            if flag:
                if count % 50 == 0:
                    center = random.choice(centers1)
                else:
                    center = random.choice(centers2)
            else:
                center = random.choice(centers2)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414  # stdev
        yield dataset


def add_nonlinearity(x, functions):
        i = 0
        for f in functions:
            i += 1
            i = i % 2
            x[:, i] = x[:, i] * f(x[:, i - 1])
        return x


def remove_nonlinearity(x, functions):
    j = len(functions)
    eps = 10 ** (-10)
    i = j % 2
    while j:
        j -= 1
        f = functions[j]
        den = f(x[:, i - 1])
        den[abs(den) < eps] = eps
        x[:, i] = x[:, i] / den
        i -= 1
        i = i % 2
    return x


def transform(x):
    # x = add_nonlinearity(x)
    x = np.vstack((x.T, np.ones(x.shape[0])))
    B = np.dot(A, x)
    B = B + np.random.normal(0, 0, B.shape)
    # B = add_nonlinearity(B)
    # B = B*B*B
    return B.T


def detransform(x):
    # x = np.cbrt(x)
    # x = remove_nonlinearity(x)
    x = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), x.T).T
    return x[:, :2]


def train(dataloader, gen, dis, functions, gen_optim, dis_optim, args):
    gen.train()
    dis.train()
    loss_func = torch.nn.BCELoss()

    n_critic = 1
    num_epoch = args.epochs
    print(num_epoch)

    for epoch in range(1, num_epoch + 1):
        for _, imgs in enumerate(dataloader):
            gen.train()
            dis.train()
            imgs = imgs.to(device)

            # Discriminatorの訓練
            dis_optim.zero_grad()

            z = torch.rand(imgs.size()[0], args.Z_dim, requires_grad=True).to(device)
            critic_fake = dis(gen(z))
            zeros = torch.zeros_like(critic_fake)

            critic_real = dis(imgs)
            ones = torch.ones_like(critic_real)

            critic_real_loss = loss_func(critic_real, ones)
            critic_fake_loss = loss_func(critic_fake, zeros)
            critic_loss = critic_real_loss + critic_fake_loss

            critic_loss.backward()
            dis_optim.step()

            # # weight clipping...
            # for p in dis.parameters():
            #   p.data.clamp_(-0.01, 0.01)

            # dis_optim.zero_grad()

            # Generatorの訓練(D1回に対してG1回)
            if epoch % n_critic == 0:
                gen_optim.zero_grad()

                z = torch.rand(ones.size()[0], args.Z_dim, requires_grad=True).to(device)

                critic_fake = dis(gen(z))
                gen_loss = loss_func(critic_fake, ones)

                gen_loss.backward()
                gen_optim.step()

        # 1000エポック毎に評価
        if epoch % 100 == 0:
            gen.eval()
            dis.eval()

            loss_dict = {
                'Discriminator': critic_loss.cpu().data.numpy(),
                'Generator': gen_loss.cpu().data.numpy()
            }
            writer.add_scalars(f'Loss', loss_dict, epoch)
            print(epoch)
            print(f"gen_loss:{gen_loss.cpu().data.numpy()}, dis_loss: {critic_loss.cpu().data.numpy()}")
            print("-" * 20)

            # plot
            z = torch.rand(args.batch_size, args.Z_dim).to(device)

            generated_points = gen(z).cpu().data.numpy()
            # print(generated_images[0])

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax2 = fig.add_subplot(111, frame_on=False)

            # realのデータ点をプロット
            reals = imgs.cpu().data.numpy()
            ax.scatter(reals[:, 0], reals[:, 1], c='r', label='real', alpha=0.5)

            # fakeのデータ点をプロット
            ax.scatter(generated_points[:, 0], generated_points[:, 1], c='b', label='fake', alpha=0.5)

            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.legend(loc='best')

            # 格子点を作成して，Dの出力を色で表示
            xp = np.linspace(-2, 2, 100)
            yp = np.linspace(-2, 2, 100)
            Dinput = []
            for i in range(len(xp)):
                for j in range(len(yp)):
                    Dinput.append([xp[i], yp[j]])
            Dinput = np.array(Dinput)

            ### ここ謎ポイント2, add_nonlinearityは何してる？ ###
            # Z = dis(torch.Tensor(add_nonlinearity(Dinput, functions)).to(device))
            Z = dis(torch.Tensor(Dinput).to(device))

            Z = Z.cpu().detach().numpy().reshape((len(xp), len(yp)))

            # ax = sns.heatmap(Z, alpha = 0.3).invert_yaxis()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            img = ax2.imshow(
                Z,
                extent=[*xlim, *ylim],
                cmap="plasma",
                aspect="auto",
                alpha=0.5,
                origin="lower",
                vmin=0.0,
                vmax=0.99,
            )
            ax2.axis('off')
            # plt.colorbar(img)

            # ax.xaxis.tick_top()
            # ax.yaxis.tick_right()

            # tensorboardにfigを追加
            writer.add_figure('Plots', fig, epoch)
            writer.flush()
            # fig.savefig(f'./plots/train_process_no-amat/epoch_{epoch}_zoom.png')


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter('./toy_no-amat_logs')
    fix_seed(42)

    # モデルの準備
    gen = Generator(args).to(device)
    dis = Discriminator(args).to(device)
    gen_optim = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    dis_optim = optim.Adam(dis.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Normalizing Flowの準備
    flows = []
    for i in range(10):
      flows.append(Flow())

    ### ここ謎ポイント1, なぜFlowの各変換を個別に持っているのか ###
    functions = []
    with torch.no_grad():
        for i in range(int(len(flows) / 2)):
            functions.append(
                lambda x: flows[i * 2](torch.FloatTensor(x).view(x.shape[0], 1))
                .view(x.shape[0])
                .detach()
                .numpy()
            )
            functions.append(
                lambda x: flows[i * 2 + 1](torch.FloatTensor(x).view(x.shape[0], 1))
                .view(x.shape[0])
                .detach()
                .numpy()
            )

    # functions.append(lambda x: x)


    # Transform matrix

    A = (np.random.rand(1000, 2 + 1) - 0.5) * 2  # keep A values in [-1,1]


    # データセットの準備
    data = gaussians(batch_size=args.batch_size, flag=False)
    points = next(data)
    points_orig = np.copy(points)
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1])
    plt.title('Original Gaussian Ring')
    plt.savefig('./plots/points_origin.png')

    # # dataset_transformed = transform(points)
    # dataset_nonlinear = add_nonlinearity(points, functions)
    # dataset_nonlinear[:, 0] = dataset_nonlinear[:, 0] * 1e5
    # dataset_nonlinear[:, 1] = dataset_nonlinear[:, 1] * 1e5
    # plt.figure()
    # plt.scatter(dataset_nonlinear[:, 0], dataset_nonlinear[:, 1])
    # plt.title('Transformed Gaussian Ring')
    # plt.savefig('./plots/points_transformed.png')
    # toy_set = toy_dataset(dataset_nonlinear)

    toy_set = toy_dataset(points)
    dataloader = DataLoader(toy_set, batch_size=args.batch_size, shuffle=True)

    train(dataloader, gen, dis, functions, gen_optim, dis_optim, args)
