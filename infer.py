import re
from pathlib import Path
import numpy as np
import torch
import scipy.stats as ss
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from models import Discriminator, Generator, ConditionalDiscriminator, ConditionalGenerator


def _load_state_dict(D, G, experiments_dir='experiments/', exper_type='1d', index=3):
    folders = [x for x in Path(experiments_dir).glob(f'{exper_type}_*') if not x.is_file()]
    folders = sorted(folders)
    experiment_dir = folders[index]

    if experiment_dir.joinpath('state_dict.pt').is_file():
        check_point = torch.load(experiment_dir.joinpath('state_dict.pt'))
        D.load_state_dict(check_point['D'])
        G.load_state_dict(check_point['G'])
    else:
        D.load_state_dict(torch.load(experiment_dir.joinpath('D.pt')))
        G.load_state_dict(torch.load(experiment_dir.joinpath('G.pt')))

    D.eval()
    G.eval()

    return D, G


def load_GAN(data_dim, z_dim, d_hidden_size, g_hidden_size, device,
             experiments_dir='experiments/', exper_type='1d', index=0):
    D = Discriminator(data_dim, d_hidden_size).to(device)
    G = Generator(z_dim, g_hidden_size, data_dim).to(device)

    _load_state_dict(D, G, experiments_dir, exper_type, index)

    return {'D': D, 'G': G}


def load_cGAN(data_dim, z_dim, d_hidden_size, g_hidden_size, n_classes, device,
              experiments_dir='experiments/', exper_type='1d', index=0):
    D = ConditionalDiscriminator(data_dim, d_hidden_size, n_classes).to(device)
    G = ConditionalGenerator(z_dim, g_hidden_size, data_dim, n_classes).to(device)

    _load_state_dict(D, G, experiments_dir, exper_type, index)

    return {'D': D, 'G': G}


def extract_wm(weights, b, projection_matrix, alpha, beta):
    weight = weights[0].mean()

    pred_bparam = weight.reshape(1, 1) @ torch.Tensor(projection_matrix)
    pred_bparam = torch.exp(alpha * torch.sin(beta * pred_bparam)) / (
            1 + torch.exp(alpha * torch.sin(beta * pred_bparam)))
    pred_bparam = pred_bparam.detach().numpy()
    pred_bparam[pred_bparam >= 0.5] = 1
    pred_bparam[pred_bparam < 0.5] = 0

    diff = np.abs(pred_bparam[0][:54] - b[:54])
    print(diff)
    print("error bits num = ", np.sum(diff.reshape(-1)))
    BER = np.sum(diff) / 54
    print("BER = ", BER)

    return pred_bparam


if __name__ == '__main__':
    '''
    _data, pdf, mu, std = load_iot_data(1)

    msg_binary = [format(ord(c), 'b').zfill(8) for c in "Taiwan"]
    msg_binary = ''.join(msg_binary)
    wm = []
    for i, b in enumerate(msg_binary):
        wm.append(int(b))
        if not ((i + 1) % 8):
            wm.append(1 if i == len(msg_binary) - 1 else 0)

    params = {
        # Parameters that we are going to tune.
        'max_depth': 9,
        'min_child_weight': 5,
        'eta': .3,
        'subsample': .7,
        'colsample_bytree': 1,
        # Other parameters
        'objective': 'binary:logistic',
        'eval_metric': 'mae'
    }

    classifier = XGBClassifier(**params)
    classifier.fit(_data[:len(wm)], wm)
    print('訓練集: ', classifier.score(_data[:len(wm)], wm))
    wm = np.concatenate([np.array(wm), classifier.predict(_data[len(wm):])])
    wm = torch.Tensor(wm).long()
    '''

    # Creating device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'\nrunning on: {device}')

    z_dim = 1
    data_dim = 1
    d_hidden_size = 30
    g_hidden_size = 15
    n_classes = 2
    exper_index = 12

    model = load_cGAN(data_dim, z_dim, d_hidden_size, g_hidden_size, n_classes, device, index=exper_index)
    D = model['D']
    G = model['G']

    data = re.compile('\r*\n').split(open('datasets/node0.txt').read())
    data = np.array(data, dtype=float)
    print(data[:8])
    scaler = MinMaxScaler()
    _data = scaler.fit_transform(data.reshape(-1, data_dim))
    mu = _data.mean()
    std = _data.std()
    pdf = ss.norm.pdf(_data, mu, std)

    folders = [x for x in Path('experiments/').glob('1d_*') if not x.is_file()]
    experiment_dir = sorted(folders)[exper_index]
    # wm = np.load(experiment_dir.joinpath('wm.npy'))
    wm = np.load('experiments/1d_gaussian_exp_1643734845/wm.npy')
    print(wm[:54])

    try:
        g_projection_matrix = np.load(experiment_dir.joinpath('g_projection_matrix.npy'))
        print('===============Generator===============')
        extract_wm(G.l3.weight, wm, g_projection_matrix, 10, 10)
    except FileNotFoundError:
        pass

    try:
        d_projection_matrix = np.load(experiment_dir.joinpath('d_projection_matrix.npy'))
        print('===============Discriminator===============')
        extract_wm(D.l3.weight, wm, d_projection_matrix, 10, 10)
    except FileNotFoundError:
        pass

    error = 0.15  # 越接近1則誤差越小
    bounded_error = 1000 * error
    z = []
    rnd_symbols = np.random.randint(2, size=len(_data))
    for i, n in enumerate(rnd_symbols):
        if n == 0:
            z.append(_data[i] + _data[i] * bounded_error)
        else:
            z.append(_data[i] + _data[i] * bounded_error)
    z = torch.tensor(z, dtype=torch.float)  # sample_noise(1000, data_dim, device)
    # z = torch.tensor(_data, dtype=torch.float)
    z_labels = torch.tensor(wm).long().to(device)  # torch.zeros(1000, 1).long().to(device)
    fake_data = G(z, z_labels).detach()

    fake_score = torch.tensor(D(fake_data, z_labels)[0], dtype=float).detach()
    pred_wm = []  # fake_score.clone()
    threshold = 0.5 * (1.11 - error)
    # threshold = 0.5 - error if error < 0.5 else error
    threshold = torch.tensor([threshold - 0.01], dtype=float)  # - 0.01 的目的是消除浮點數運算，避免數值一樣時判斷錯誤
    for item in fake_score.reshape(-1, 4):
        temp = sum(np.less(item, threshold))
        pred_wm.append(0 if np.round(temp / 4 + 0.1) else 1)
    '''
    for i, b in enumerate(pred_wm):
        if torch.lt(b, threshold):
            pred_wm[i] = 1
        else:
            pred_wm[i] = 0
    '''
    pred_wm = torch.tensor(pred_wm)
    print('===============Predicted WM===============')
    print(pred_wm[:54].reshape(-1))

    diff = np.abs(pred_wm.long()[:54] - wm.reshape(-1, 1))
    print("error bits num = ", torch.sum(diff.reshape(-1)).item())
    BER = torch.sum(diff) / 54
    print("BER = ", BER.item())

    print('===============Generated data===============')
    with open('./datasets/generated.txt', 'w') as writer:
        t_fake_data = (fake_data - fake_data.min()) / (fake_data.max() - fake_data.min())
        generated_data = scaler.inverse_transform(t_fake_data).reshape(-1)
        generated_data = np.round(generated_data).astype(int)
        print(t_fake_data[:8].reshape(-1))
        print(generated_data[:8])

        writer.write('\r\n'.join(map(str, generated_data)))

    fake_score = round(torch.mean(fake_score).item(), 5)
    f_mean = round(torch.mean(fake_data).item(), 4)
    f_std = round(torch.std(fake_data).item(), 4)
    stats = f'\nGenerated data stats (mean, std): {f_mean}, {f_std}\nfake score: {fake_score}'
    print(stats)

    # classifier = SVC().fit(data.reshape(-1, 1)[:54], wm[:54])
    # print(classifier.predict(generated_data.reshape(-1, 1)))

    # plt.title(f'Real and Synthetic data (bounded-error: {error * 1000}%)')
    plt.plot(_data, pdf, color="green", label="Original data", alpha=0.5)
    plt.plot(t_fake_data, ss.norm.pdf(t_fake_data, t_fake_data.mean(), t_fake_data.std()), color="red",
             label="Generated data", alpha=0.5)
    plt.legend(loc='upper right')
    # plt.ylim(800, 2400)
    plt.show()

    # plt.savefig('smartmeter_generated.png')