import numpy as np
import torch
import torch.nn.functional as F


class STDM:
    def __init__(self, b, scale, alpha=10, beta=10):
        self.scale = scale
        self.b = b
        self.alpha = alpha
        self.beta = beta

    def __call__(self, x):
        pro_matrix_rows = np.prod(x.shape[0])
        pro_matrix_cols = self.b.shape[0]

        regularization = 0

        w_mean = torch.mean(x, axis=len(x.shape) - 1)
        try:
            w_mean = torch.reshape(w_mean, (1, pro_matrix_rows))
            self.pro_matrix_value = np.random.randn(pro_matrix_rows, pro_matrix_cols)
        except:
            w_mean = torch.reshape(w_mean, (1, 1))
            self.pro_matrix_value = np.random.randn(1, pro_matrix_cols)
        projection_matrix = torch.tensor(self.pro_matrix_value)
        wm_projection_matrix = w_mean.float() @ projection_matrix.float()

        alpha = self.alpha
        beta = self.beta

        regularization += self.scale * torch.sum(
            F.binary_cross_entropy(
                torch.exp(alpha * torch.sin(beta * wm_projection_matrix)) / (
                        1 + torch.exp(alpha * torch.sin(beta * wm_projection_matrix))),
                self.b.reshape(1, -1).float()
            ))

        return regularization

    def save_projection_matrix(self, pro_matrix_path):
        np.save(pro_matrix_path, self.pro_matrix_value)
