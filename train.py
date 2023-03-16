import argparse
import ast
import re
from pathlib import Path
from time import time

import numpy as np
import matplotlib
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scipy.stats as ss
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

import torch

from dest._sample import (sample_noise, sample_1d_data)
from models import (ConditionalDiscriminator, ConditionalGenerator)
from losses import (conditional_d_loss, conditional_g_loss)
from utility import (clear_line, clear_patch)
from wm_regularization import STDM


matplotlib.use('Qt5Agg')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=15000)
    parser.add_argument('--minibatch', type=int, default=50)
    parser.add_argument('--d-lr', type=float, default=0.0002)
    parser.add_argument('--g-lr', type=float, default=0.0002)
    parser.add_argument('--d-optim', type=str, default='adam')
    parser.add_argument('--g-optim', type=str, default='adam')
    parser.add_argument('--loss', type=str, default='hack')
    parser.add_argument('--d-hidden', type=int, default=30)
    parser.add_argument('--g-hidden', type=int, default=15)
    parser.add_argument('--pui', type=int, default=1)
    parser.add_argument('--psi', type=int, default=1000)
    parser.add_argument('--save-figs', type=ast.literal_eval, default='True')
    parser.add_argument('--save-model', type=ast.literal_eval, default='True')
    parser.add_argument('--msg', type=str, required=True)

    args = vars(parser.parse_args())
    default = {
        'z_dim': 1,
        'data_dim': 1,
        'n_classes': 2,
    }

    return {
        **default,
        **args
    }


def load_sample_data(data_dim):
    mu = 2.0
    std = 0.75
    _data = sample_1d_data(200, data_dim, torch.device('cpu'), mu, std).numpy()
    _data.sort(axis=0)
    pdf = ss.norm.pdf(_data, mu, std)

    return _data, pdf, mu, std


def load_iot_data(data_dim):
    _data = re.compile('\r*\n').split(open('datasets/node0.txt').read())
    _data = np.array(_data, dtype=float)
    _data = MinMaxScaler().fit_transform(_data.reshape(-1, data_dim))
    mu = _data.mean()
    std = _data.std()
    pdf = ss.norm.pdf(_data, mu, std)

    return _data, pdf, mu, std


def main():
    # config experiment
    expr_config = parse_args()

    num_epochs = expr_config['epochs']
    minibatch_size = expr_config['minibatch']
    d_learning_rate = expr_config['d_lr']
    g_learning_rate = expr_config['g_lr']
    discriminator_optim = expr_config['d_optim']
    generator_optim = expr_config['g_optim']
    loss_type = expr_config['loss']
    z_dim = expr_config['z_dim']
    data_dim = expr_config['data_dim']
    d_hidden_size = expr_config['d_hidden']
    g_hidden_size = expr_config['g_hidden']
    n_classes = expr_config['n_classes']
    progress_update_interval = expr_config['pui']
    progress_save_interval = expr_config['psi']
    msg = expr_config['msg']
    save_figs = expr_config['save_figs']
    save_model = expr_config['save_model']

    experiment_info = f'\ntotal iters: {num_epochs},\nbatch_size: {minibatch_size},\nd_lr: {d_learning_rate},\n' + \
                      f'g_lr: {g_learning_rate},\nloss: {loss_type},\nd_hidden_size: {d_hidden_size},\n' + \
                      f'g_hidden_size: {g_hidden_size},\ndisc_optim: {discriminator_optim},\n' + \
                      f'gen_optim: {generator_optim},\ndata_dim: {data_dim},\nz_dim: {z_dim},\n' + \
                      f'random seed: {SEED}'
    print(experiment_info)

    # Create experiment folders(if necessary).
    experiment_dir = Path(f'./experiments/1d_gaussian_exp_{time():.0f}')
    if save_model:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        fig_shots_dir = experiment_dir.joinpath('graph_shots')
        if save_figs:
            fig_shots_dir.mkdir(parents=True, exist_ok=True)

    # Create experiment info file(if necessary).
    info_file = None
    if save_model:
        info_file = open(experiment_dir.joinpath('info.txt'), mode='w', encoding='utf-8')
        info_file.write(experiment_info)

    # plots
    # _data, pdf, mu, std = load_sample_data(data_dim)
    _data, pdf, mu, std = load_iot_data(data_dim)

    fig, ax_data, ax_loss, ax_disc = prepare_plots(_data, pdf, experiment_info, '1D Gaussian')
    plt.ion()
    plt.show()

    msg_binary = [format(ord(c), 'b').zfill(8) for c in msg]
    msg_binary = ''.join(msg_binary)
    wm = []
    n_repeat = 4
    for i, b in enumerate(msg_binary):
        wm += n_repeat * [int(b)]
        if not ((i + 1) % 8):
            divided_flag = 1 if i == len(msg_binary) - 1 else 0
            wm += n_repeat * [divided_flag]

    classifier = KNeighborsClassifier(n_neighbors=1).fit(_data[:len(wm)], wm)
    wm = np.concatenate([np.array(wm), classifier.predict(_data[len(wm):])])
    wm = torch.Tensor(wm).long()

    '''
    classifier = XGBClassifier()
    classifier.fit(_data[:len(wm)], wm)
    wm = np.concatenate([np.array(wm), classifier.predict(_data[len(wm):])])
    wm = torch.Tensor(wm).long()
    '''

    # Creating device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'\nrunning on: {device}')

    # Creating models
    # D = Discriminator(data_dim, d_hidden_size).to(device)
    # G = Generator(z_dim, g_hidden_size, data_dim).to(device)
    D = ConditionalDiscriminator(data_dim, d_hidden_size, n_classes).to(device)
    G = ConditionalGenerator(z_dim, g_hidden_size, data_dim, n_classes).to(device)
    print(f'\n\n{D}\n\n{G}\n\n')

    if save_model:
        info_file.write(f'\nrunning on: {device}')
        info_file.write(f'\n\n{D}\n\n{G}\n\n\n')
        info_file.flush()

    if discriminator_optim == 'sgd':
        d_optimizer = torch.optim.SGD(D.parameters(), lr=d_learning_rate, momentum=0.65)
    else:
        d_optimizer = torch.optim.Adam(D.parameters(), lr=d_learning_rate, betas=(0.5, 0.999))

    if generator_optim == 'sgd':
        g_optimizer = torch.optim.SGD(G.parameters(), lr=g_learning_rate, momentum=0.65)
    else:
        g_optimizer = torch.optim.Adam(G.parameters(), lr=g_learning_rate, betas=(0.5, 0.999))

    show_separate_loss = False
    d_real_loss_list: list = []
    d_fake_loss_list: list = []
    g_loss_list: list = []
    d_x_list: list = []
    d_g_z_list: list = []

    d_regularization = STDM(wm, 0.01)
    g_regularization = STDM(wm, 0.01)

    #
    # training loop
    #
    t1 = time()
    for epoch in range(1, num_epochs + 1):
        real_label = torch.ones(minibatch_size, 1).to(device)
        fake_label = torch.zeros(minibatch_size, 1).to(device)

        # Training discriminator
        for k in range(1):
            # real_data = sample_1d_data(minibatch_size, data_dim, device, mu, std)

            df = pd.DataFrame({
                'data': _data.reshape(-1),
                'label': wm
            }).sample(minibatch_size)
            real_data = torch.Tensor(df['data'].to_numpy()).reshape(-1, 1)
            labels = torch.Tensor(df['label'].to_numpy()).unsqueeze(
                1).long()
            # labels = torch.Tensor(classifier.predict(real_data)).unsqueeze(1).long()
            real_score, _ = D(real_data, labels)
            real_loss = conditional_d_loss(real_score, real_label)

            z_noise = sample_noise(minibatch_size, z_dim, device)
            d_fake_data = G(z_noise, labels).detach()  # detach to avoid training G on these data.
            d_fake_score, _ = D(d_fake_data, labels)
            fake_loss = conditional_d_loss(d_fake_score, fake_label)

            d_loss = (real_loss + fake_loss) / 2
            # d_loss += sum(d_regularization(p) for p in D.parameters())
            # d_regularization.save_projection_matrix(experiment_dir.joinpath('d_projection_matrix.npy'))

            # d_loss, real_loss, fake_loss = standard_d_loss(real_score, d_fake_score)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

        # Training generator
        for g in range(1):
            # g_z_noise = sample_noise(minibatch_size, z_dim, device)
            fake_data = G(z_noise, labels)
            fake_score, _ = D(fake_data, labels)  # this is D(G(z))

            g_loss = conditional_g_loss(fake_score, real_label)
            g_loss += sum(g_regularization(p) for p in G.parameters())
            g_regularization.save_projection_matrix(experiment_dir.joinpath('g_projection_matrix.npy'))

            # if loss_type == 'ce':
            #     g_loss = standard_g_loss(fake_score, real_score.detach())
            # elif loss_type == 'hack':
            #     g_loss = heuristic_g_loss(fake_score)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        # Show some progress!
        if epoch % progress_update_interval == 0 or epoch == num_epochs:
            loss_log = f'epoch #{epoch:<5}:\n\tD_loss: {d_loss:<10.5f}, G_loss: {g_loss:<10.5f}' + \
                       f'\n\tD scores:  real: {torch.mean(real_score):.5f}\t' + \
                       f'fake: {torch.mean(d_fake_score):.5f}\t  G score: {torch.mean(fake_score):.5f}\n'

            print(loss_log)
            if save_model:
                info_file.write(loss_log)
                info_file.flush()

            if show_separate_loss:
                d_real_loss_list.append(real_loss.tolist())
                d_fake_loss_list.append(fake_loss.tolist())
            else:
                d_real_loss_list.append(d_loss.tolist())
            #
            g_loss_list.append(g_loss.tolist())

            # plot losses
            update_loss_plot(
                ax_loss, d_real_loss_list, d_fake_loss_list, g_loss_list,
                progress_update_interval, show_separate_loss
            )

            # plot d(x) and d(g(z))
            d_x_list.append(torch.mean(real_score).item())
            d_g_z_list.append(torch.mean(fake_score).item())
            update_disc_plot(ax_disc, d_x_list, d_g_z_list, progress_update_interval)

            # plot generated data
            z = sample_noise(1000, z_dim, device)
            # TODO: z_labels 需要去測試要用 1 或是 真實數據的類別標籤
            z_labels = wm.to(device)  # torch.Tensor(classifier.predict(z)).unsqueeze(1).long()
            fake_data = G(z, z_labels).detach().cpu().numpy()
            update_data_plot(ax_data, D, fake_data, z_labels, device)

            # Refresh figure
            fig.canvas.draw()
            fig.canvas.flush_events()

            # if save_model and save_figs:
            #     f = fig_shots_dir.joinpath(f'shot_{epoch // progress_update_interval}.png')
            #     fig.savefig(f, format='png')

        if save_figs and (epoch % progress_save_interval == 0 or epoch == num_epochs):
            f = fig_shots_dir.joinpath(f'shot_{epoch // progress_save_interval}.png')
            fig.savefig(f, format='png')

    # End of training
    t2 = time()
    elapsed = round((t2 - t1))
    minutes, seconds = divmod(elapsed, 60)
    elapsed = f'\n\nelapsed time: {minutes:02d}:{seconds:02d}'
    print(elapsed)

    z = sample_noise(1000, z_dim, device)
    # TODO: z_labels 需要去測試要用 1 或是 真實數據的類別標籤
    z_labels = wm.to(device)  # torch.Tensor(classifier.predict(z)).unsqueeze(1).long()
    fake_data = G(z, z_labels).detach()
    fake_score = round(torch.mean(D(fake_data, z_labels)[0].detach()).item(), 5)
    f_mean = round(torch.mean(fake_data).item(), 4)
    f_std = round(torch.std(fake_data).item(), 4)
    stats = f'\nGenerated data stats (mean, std): {f_mean}, {f_std}\nfake score: {fake_score}'
    print(stats)

    if save_model:
        info_file.write(elapsed)
        info_file.write(stats)
        info_file.flush()
        info_file.close()

        torch.save({
            'G': G.state_dict(),
            'D': D.state_dict(),
            'G_optim': g_optimizer.state_dict(),
            'D_optim': d_optimizer.state_dict(),
        }, experiment_dir.joinpath('state_dict.pt'))
        torch.save(G.state_dict(), experiment_dir.joinpath('G.pt'))
        torch.save(D.state_dict(), experiment_dir.joinpath('D.pt'))

    input('\n\npress Enter to end...\n')


def decay_lr(d_optimizer, g_optimizer):
    for param_group in d_optimizer.param_groups:
        param_group['lr'] *= 0.999

    for param_group in g_optimizer.param_groups:
        param_group['lr'] *= 0.999


def prepare_plots(data, pdf, info, title=''):
    fig: plt.Figure = plt.figure(1, figsize=(14, 8.0))
    fig.canvas.manager.set_window_title(title)

    ax_data = fig.add_subplot(3, 1, 1)
    ax_loss = fig.add_subplot(3, 1, 2)
    ax_disc = fig.add_subplot(3, 1, 3)

    fig.tight_layout(h_pad=1.55, rect=[0.01, 0.04, 0.99, 0.98])

    ax_data.set_title('Real v.s. Generated', fontweight='bold')
    ax_data.plot(
        data,
        pdf,
        label='data',
        color='royalblue',
        marker='.',
        markerfacecolor='navy',
        markeredgecolor='darkmagenta',
        linestyle='solid',
        linewidth=4,
        markersize=7
    )
    # ax_data.set_xlim([-1.5, 5.5])
    ax_data.set_xlim([-0.5, 1.5])
    # ax_data.set_ylim([0, 1.03])
    ax_data.set_ylim([data.reshape(1, -1).min(), data.reshape(1, -1).max() + 2])
    ax_data.annotate(
        info.replace('\n', '  '),
        xy=(0, 0),
        xytext=(2, 14),
        xycoords=('figure pixels', 'figure pixels'),
        textcoords='offset pixels',
        bbox=dict(facecolor='dodgerblue', alpha=0.15),
        size=9.5,
        ha='left'
    )

    ax_loss.set_title('Losses', fontweight='bold')
    ax_loss.grid()

    ax_disc.set_title('Discriminator Outputs', fontweight='bold')
    ax_disc.grid()

    return fig, ax_data, ax_loss, ax_disc


def update_loss_plot(ax: plt.Axes, d_loss, d_fake_loss, g_loss, update_interval, separate=False):
    clear_line(ax, 'd_loss')
    clear_line(ax, 'g_loss')

    x = np.arange(1, len(d_loss) + 1)

    if separate:
        ax.plot(x, np.add(d_loss, d_fake_loss), color='dodgerblue', label='D Loss', gid='d_loss')
        clear_line(ax, 'd_real_loss')
        ax.plot(x, d_loss, color='lightseagreen', label='D Loss(Real)', gid='d_real_loss')
        clear_line(ax, 'd_fake_loss')
        ax.plot(x, d_fake_loss, color='mediumpurple', label='D Loss(Fake)', gid='d_fake_loss')
    else:
        ax.plot(x, d_loss, color='dodgerblue', label='D Loss', gid='d_loss')

    ax.plot(x, g_loss, color='coral', label='G Loss', gid='g_loss', alpha=0.9)
    ax.legend(loc='upper right', framealpha=0.75)
    ax.set_xlim(left=1, right=len(x) + 0.01)
    ticks = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks * update_interval))
    ax.set_xticklabels([f'{t:.0f}' for t in ticks * update_interval])


def update_disc_plot(ax: plt.Axes, d_x, d_g_z, update_interval):
    clear_line(ax, 'dx')
    clear_line(ax, 'dgz')

    x = np.arange(1, len(d_x) + 1)
    ax.plot(x, d_x, color='#308862', label='D(x)', gid='dx')
    ax.plot(x, d_g_z, color='#B23F62', label='D(G(z))', gid='dgz', alpha=0.9)
    ax.legend(loc='upper right', framealpha=0.75)
    ax.set_xlim(left=1, right=len(x) + 0.01)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1, 0.1))
    ticks = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks * update_interval))
    ax.set_xticklabels([f'{t:.0f}' for t in (ticks * update_interval)])


def update_data_plot(ax, D, fake_data, labels, device):
    # draw decision discriminator boundary
    clear_line(ax, 'decision')
    plot_decision_boundary(ax, D, labels, device)
    #
    clear_patch(ax, 'g_hist')
    ax.hist(
        fake_data,
        gid='g_hist',
        bins=100,
        density=True,
        histtype='stepfilled',
        label='generated',
        facecolor='moccasin',
        edgecolor='sandybrown',
        linewidth=2,
        alpha=0.85
    )
    ax.legend(loc='upper right', framealpha=0.75)


def plot_decision_boundary(ax: plt.Axes, discriminator, labels, device=torch.device('cpu')) -> None:
    _data = torch.linspace(-5, 9, 1000, requires_grad=False).view(1000, 1).to(device)
    decision = discriminator(_data, labels)
    if type(decision) == tuple:
        decision = decision[0]

    ax.plot(
        _data.cpu().numpy(),
        decision.detach().cpu().numpy(),
        gid='decision',
        label='decision boundary',
        color='gray',
        linestyle='--'
    )


if __name__ == "__main__":
    main()
