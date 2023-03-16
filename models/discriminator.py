import torch
from torch import nn
import torch.nn.functional as F
from torch.tensor import Tensor
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, data_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(data_dim, hidden_size),
            nn.ELU(alpha=0.05)
            # nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(alpha=0.05)
            # nn.ReLU()
        )
        self.l3 = torch.nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        out = self.l1(x)
        feat = self.l2(out)
        out = self.l3(feat)
        out = self.sigmoid(out)

        return out, feat


class Discriminator8G(nn.Module):
    def __init__(self, data_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(data_dim, hidden_size),
            nn.ELU(alpha=0.05)
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ELU(alpha=0.05)
        )
        self.l3 = torch.nn.Linear(hidden_size // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.sigmoid(out)

        return out


class Discriminator25G(nn.Module):
    def __init__(self, data_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(data_dim, hidden_size),
            nn.ELU(alpha=0.05)
            # nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(alpha=0.05)
            # nn.ReLU()
        )
        self.l3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ELU(alpha=0.05)
            # nn.ReLU()
        )
        self.l4 = nn.Linear(hidden_size // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.sigmoid(out)

        return out


class DiscriminatorMixture(nn.Module):
    def __init__(self, data_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(data_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size * 2)
        self.l3 = nn.Linear(hidden_size * 2, hidden_size)
        self.l4 = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        out = torch.sigmoid(self.l4(out))

        return out


class DiscriminatorMNIST(nn.Module):
    def __init__(self, data_dim: int, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(data_dim, hidden_size),
            nn.ELU(alpha=0.2),
            nn.Dropout(0.3)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.fc1[0].out_features, self.fc1[0].out_features // 2),
            nn.ELU(alpha=0.2),
            nn.Dropout(0.3)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(self.fc2[0].out_features, self.fc2[0].out_features // 2),
            nn.ELU(alpha=0.2),
            nn.Dropout(0.3)
        )
        self.fc4 = nn.Linear(self.fc3[0].out_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.sigmoid(out)

        return out


class ConditionalDiscriminator(nn.Module):
    def __init__(self, data_dim: int, hidden_size: int, n_classes: int) -> None:
        super().__init__()
        self.label_embedding = nn.Sequential(
            nn.Embedding(n_classes, data_dim),
            nn.Linear(data_dim, data_dim)
        )
        self.l1 = nn.Sequential(
            nn.Linear(data_dim + n_classes - 1, hidden_size),
            nn.ELU(alpha=0.05)
            # nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(alpha=0.05)
            # nn.ReLU()
        )
        self.l3 = torch.nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def _embed(self, x, m):
        """
        x is a vector of values to be quantized individually
        m is a binary vector of bits to be embeded
        returns: a quantized vector y
        """
        x = x.astype(float)
        d = 10
        y = np.round(x / d) * d + (-1) ** (m + 1) * d / 4.
        return y

    def _detect_wm(self, z):

        """
        z is the received vector, potentially modified
        returns: a detected vector z_detected and a detected message m_detected
        """

        z = z.detach().numpy()
        shape = z.shape
        z = z.flatten()

        m_detected = np.zeros_like(z, dtype=float)
        z_detected = np.zeros_like(z, dtype=float)

        z0 = self._embed(z, 0)
        z1 = self._embed(z, 1)

        d0 = np.abs(z - z0)
        d1 = np.abs(z - z1)

        gen = zip(range(len(z_detected)), d0, d1)
        for i, dd0, dd1 in gen:
            if dd0 < dd1:
                m_detected[i] = 0
                z_detected[i] = z0[i]
            else:
                m_detected[i] = 1
                z_detected[i] = z1[i]

        z_detected = z_detected.reshape(shape)
        m_detected = m_detected.reshape(shape)
        return torch.Tensor(m_detected.astype(int))

    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        labels = self.label_embedding(labels)
        labels = labels.view(-1, 1)
        out = torch.cat((x, labels), dim=1)
        out = self.l1(out)
        feat = self.l2(out)
        out = self.l3(feat)
        out = self.sigmoid(out)
        # out = self._detect_wm(out)

        return out, feat


class WmLayer(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

    def _embed(self, x, m):
        """
        x is a vector of values to be quantized individually
        m is a binary vector of bits to be embeded
        returns: a quantized vector y
        """
        x = x.astype(float)
        d = 10
        y = np.round(x / d) * d + (-1) ** (m + 1) * d / 4.
        return y

    def forward(self, x):
        x = x.detach().numpy()
        shape = x.shape
        z = x.flatten()

        m_detected = np.zeros_like(z, dtype=float)
        z_detected = np.zeros_like(z, dtype=float)

        z0 = self._embed(z, 0)
        z1 = self._embed(z, 1)

        d0 = np.abs(z - z0)
        d1 = np.abs(z - z1)

        gen = zip(range(len(z_detected)), d0, d1)
        for i, dd0, dd1 in gen:
            if dd0 < dd1:
                m_detected[i] = 0
                z_detected[i] = z0[i]
            else:
                m_detected[i] = 1
                z_detected[i] = z1[i]

        z_detected = z_detected.reshape(shape)
        m_detected = m_detected.reshape(shape)
        return m_detected.astype(int)
