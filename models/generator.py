import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.tensor import Tensor

from ._lstm import NaiveLSTM


class Generator(nn.Module):
    def __init__(self, z_dim: int, hidden_size: int, data_dim: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.ELU(alpha=0.2)
            # nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(alpha=0.2)
            # nn.ReLU()
        )
        self.l3 = nn.Linear(hidden_size, data_dim)

    def forward(self, x: Tensor) -> Tensor:
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)

        return out


class Generator8G(nn.Module):
    def __init__(self, z_dim: int, hidden_size: int, data_dim: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.ELU(alpha=0.2)
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(alpha=0.2)
        )
        self.l3 = nn.Linear(hidden_size, data_dim)

    def forward(self, x: Tensor) -> Tensor:
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)

        return out


class Generator25G(nn.Module):
    def __init__(self, z_dim: int, hidden_size: int, data_dim: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.ELU(alpha=0.2)
            # nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(alpha=0.2)
            # nn.ReLU()
        )
        self.l3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(alpha=0.2)
            # nn.ReLU()
        )
        self.l4 = nn.Linear(hidden_size, data_dim)

    def forward(self, x: Tensor) -> Tensor:
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)

        return out


class GeneratorMixture(nn.Module):
    def __init__(self, z_dim: int, hidden_size: int, data_dim: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(z_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size * 2)
        self.l3 = nn.Linear(hidden_size * 2, hidden_size)
        self.l4 = nn.Linear(hidden_size, data_dim)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))

        return self.l4(out)


class GeneratorMNIST(nn.Module):
    def __init__(self, z_dim: int, hidden_size: int, data_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.ELU(alpha=0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.fc1[0].out_features, self.fc1[0].out_features * 2),
            nn.ELU(alpha=0.2)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(self.fc2[0].out_features, self.fc2[0].out_features * 2),
            nn.ELU(alpha=0.2)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(self.fc2[0].out_features * 2, data_dim),
            nn.Tanh()
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)

        return out


class ConditionalGenerator(nn.Module):
    def __init__(self, z_dim: int, hidden_size: int, data_dim: int, n_classes: int) -> None:
        super().__init__()
        self.label_embedding = nn.Sequential(
            nn.Embedding(n_classes, z_dim),
            nn.Linear(z_dim, z_dim)
        )
        self.l1 = nn.Sequential(
            nn.Linear(z_dim + n_classes - 1, hidden_size),
            nn.ELU(alpha=0.2)
            # nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(alpha=0.2)
            # nn.ReLU()
        )
        self.l3 = nn.Linear(hidden_size, data_dim)

    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        labels = self.label_embedding(labels)
        labels = labels.view(-1, 1)
        out = torch.cat((x, labels), dim=1)
        out = self.l1(out)
        out = self.l2(out)
        out = self.l3(out)

        return out


class CGanWithLSTM(ConditionalGenerator):
    def __init__(self, z_dim: int, hidden_size: int, data_dim: int, n_classes: int) -> None:
        super(CGanWithLSTM, self).__init__(z_dim, hidden_size, data_dim, n_classes)

        self.lstm = NaiveLSTM(z_dim, z_dim)
        self.l1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(alpha=0.2)
            # nn.ReLU()
        )

    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        labels = self.label_embedding(labels)
        labels = labels.view(-1, 1)
        bs, ds = x.size()
        out, (h_t, c_t) = self.lstm(x.reshape(bs // 2, 2, ds), bound=0.01)
        out = torch.cat((out, labels), dim=1)
        out = self.l1(out)
        out = self.l2(out)
        out = self.l3(out)

        return out
