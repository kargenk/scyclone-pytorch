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


def get_critic_acc(critic_fake, critic_real):
    acc = 0.0
    for x in critic_fake[:]:
        if x.item() <= 0.5:
            acc += 1
    for x in critic_real[:]:
        if x.item() >= 0.5:
            acc += 1
    acc /= (critic_fake.size()[0] + critic_real.size()[0])
    return acc


def d_train(x_real, G, multi_D, multi_D_optim, num_D, args):
    for i in range(num_D):
        multi_D[i].train()
        for p in multi_D[i].parameters():
            p.requires_grad = True
        multi_D_optim[i].zero_grad()

    flag = True
    z = torch.randn(x_real.size()[0], args.Z_dim, requires_grad=True).to(device)
    # z = Variable(torch.randn(x_real.size()[0], z_dim, 1, 1).to(device))
    x_fake = G(z)
    for i in range(num_D):
        if flag:
            D_fake = multi_D[i](x_fake).unsqueeze(1)
            D_real = multi_D[i](x_real).unsqueeze(1)
            flag = False
        else:
            D_fake = torch.cat((D_fake, multi_D[i](x_fake).unsqueeze(1)), dim=1)
            D_real = torch.cat((D_real, multi_D[i](x_real).unsqueeze(1)), dim=1)

    superior_idx = torch.argmin(D_fake, dim=1)  # 出力スコアが最も低い(fakeをfakeと判定している)Discriminatorを選択
    mask = torch.zeros((x_real.size()[0], num_D)).to(device)

    # 1バッチ毎にε:0.3で優秀なD,1-εでランダムなDを選択
    for i in range(mask.size()[0]):
        random_checker = np.random.randint(0, 10)
        if random_checker > 7:
            index = np.random.randint(0, num_D)
            mask[i][index] = 1.0
        else:
            mask[i][superior_idx[i]] = 1.0

    D_fake = D_fake.squeeze()
    D_real = D_real.squeeze()
    D_fake_output = torch.sum(mask * D_fake, dim=1)
    D_real_output = torch.sum(mask * D_real, dim=1)

    y_real = torch.ones_like(D_real_output, requires_grad=False)
    y_fake = torch.zeros_like(D_fake_output, requires_grad=False)

    D_real_loss = criterion(D_real_output, y_real)
    D_fake_loss = criterion(D_fake_output, y_fake)

    D_acc = get_critic_acc(D_fake_output, D_real_output)

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    for i in range(num_D):
        if i in superior_idx:
            multi_D_optim[i].step()

    return  D_loss.data.item(), D_acc, superior_idx


def g_train(x_real, G, G_optim, multi_D, num_D, args):
    G.train()
    G_optim.zero_grad()

    z = torch.randn(x_real.size()[0], args.Z_dim, requires_grad=True).to(device)

    critic_fakes = []
    fake_img = G(z)
    lit = np.zeros(num_D)
    for i in range(num_D):
        for p in multi_D[i].parameters():
            p.requires_grad = False
        critic_fake = multi_D[i](fake_img)
        critic_fakes.append(critic_fake)
        lit[i] = torch.sum(critic_fake).item()
    # 再重み付け
    loss_sort = np.argsort(lit)
    weights = np.random.dirichlet(np.ones(num_D))
    weights = np.sort(weights)[::-1]

    flag = False
    for i in range(len(critic_fakes)):
        if flag == False:
            critic_fake = weights[i] * critic_fakes[loss_sort[i]]
            flag = True
        else:
            critic_fake = torch.add(critic_fake, weights[i] * critic_fakes[loss_sort[i]])

    y = torch.ones_like(critic_fake, requires_grad=False)
    G_loss = criterion(critic_fake, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optim.step()

    return G_loss.data.item()


def train(dataloader, G, multi_D, functions, G_optim, multi_D_optim, num_D, args):
    n_critic = 1
    num_epoch = args.epochs
    print(num_epoch)

    for epoch in range(1, num_epoch + 1):
        G.train()
        D_losses, D_accs, G_losses = [], [], []
        for batch_idx, x_real in enumerate(dataloader):
            x_real = x_real.to(device)
            D_loss, D_acc, d_idx = d_train(x_real, G, multi_D, multi_D_optim, num_D, args)
            D_losses.append(D_loss)
            D_accs.append(D_acc)
            if batch_idx % n_critic == 0:
                G_losses.append(g_train(x_real, G, G_optim, multi_D, num_D, args))

        # 1000epochごとに評価
        if epoch % 100 == 0:
            print(f'[{epoch}/{num_epoch+1}]: loss_d: {np.mean(D_losses)}, acc_d: {np.mean(D_accs)}, loss_g: {np.mean(G_losses)}')
            print(f'\ndis_index current: {d_idx}')

            # TensorBoardにログを保存
            loss_dict = {
                'Discriminator': np.mean(D_losses),
                'Generator': np.mean(G_losses),
            }
            writer.add_scalars(f'Loss', loss_dict, epoch)

            acc_dict = {'Discriminator': np.mean(D_accs),}
            writer.add_scalars(f'Accuracy', acc_dict, epoch)

            # plot
            z = torch.randn(args.batch_size, args.Z_dim).to(device)
            generated_points = G(z).cpu().data.numpy()

            fig = plt.figure()
            ax = fig.add_subplot(111)

            # realのデータ点をプロット
            reals = x_real.cpu().data.numpy()
            ax.scatter(reals[:, 0], reals[:, 1], c='r', label='real', alpha=0.5)

            # fakeのデータ点をプロット
            ax.scatter(generated_points[:, 0], generated_points[:, 1], c='b', label='fake', alpha=0.5)
            print(generated_points[:5])

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
            # d_out = dis(torch.Tensor(add_nonlinearity(Dinput, functions)).to(device))
            for i in range(num_D):
                d_out = multi_D[i](torch.Tensor(Dinput).to(device))

                d_out = d_out.cpu().detach().numpy().reshape((len(xp), len(yp)))

                # ax = sns.heatmap(Z, alpha = 0.3).invert_yaxis()
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                ax2 = fig.add_subplot(111, frame_on=False)
                img = ax2.imshow(
                    d_out,
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
                writer.add_figure(f'Plots/D{i}', fig, epoch)
                writer.flush()
                del ax2
                # fig.savefig(f'./plots/train_process_no-amat/epoch_{epoch}_zoom.png')


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter('./logs/toy_amat_logs')
    fix_seed(42)

    # モデルと最適化手法
    G = Generator(args).to(device)
    G_optim = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))

    num_D = 3
    multi_D = []
    multi_D_optim = []
    for i in range(num_D):
        multi_D.append(Discriminator(args).to(device))
        multi_D_optim.append(optim.Adam(multi_D[i].parameters(), lr=args.lr, betas=(0.5, 0.999)))

    criterion = nn.BCELoss()

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
    plt.savefig('./plots/amat/points_origin.png')

    # # dataset_transformed = transform(points)
    # dataset_nonlinear = add_nonlinearity(points, functions)
    # dataset_nonlinear[:, 0] = dataset_nonlinear[:, 0] * 1e4
    # dataset_nonlinear[:, 1] = dataset_nonlinear[:, 1] * 1e5
    # plt.figure()
    # plt.scatter(dataset_nonlinear[:, 0], dataset_nonlinear[:, 1])
    # plt.title('Transformed Gaussian Ring')
    # plt.savefig('./plots/amat/points_transformed.png')
    # toy_set = toy_dataset(dataset_nonlinear)

    toy_set = toy_dataset(points)
    dataloader = DataLoader(toy_set, batch_size=args.batch_size, shuffle=True)

    train(dataloader, G, multi_D, functions, G_optim, multi_D_optim, num_D, args)
