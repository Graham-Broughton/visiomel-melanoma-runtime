import torch
import torch.nn.functional as F
import torch.nn as nn
class Meta(nn.Module):
    def __init__(self, input_shape, l_nodes=None, dropout_pct=0.2):
        if l_nodes is None:
            l_nodes = [256, 256]
        super(Meta, self).__init__()
        self.l1 = nn.Linear(input_shape, l_nodes[0])
        self.l2 = nn.Linear(l_nodes[0], l_nodes[1])

        self.bn1 = nn.BatchNorm1d(l_nodes[0])
        self.bn2 = nn.BatchNorm1d(l_nodes[1])

        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=dropout_pct)

    def forward(self, x_meta):
        x = self.l1(x_meta)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dp(x)
        x = self.dp(self.relu(self.bn2(self.l2(x))))
        return x

class Attention(nn.Module):
    def __init__(self, input_D: int = 128, out_kernels_1: int = 36, out_kernels_2: int = 48, L: int = 512, K: int = 1, meta_model = Meta):
        super(Attention, self).__init__()
        self.L = L  # 512 node fully connected layer
        self.output_D = ((((input_D - 3) // 2) - 4) // 2) + 1
        self.kernel_1 = out_kernels_1
        self.kernel_2 = out_kernels_2
        self.D = input_D  # 128 node attention layer
        self.K = K
        self.meta_model = meta_model(3)

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, self.kernel_1, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(self.kernel_1, self.kernel_2, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(self.kernel_2 * self.output_D * self.output_D, self.L),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            nn.Dropout()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_meta):
        x = x.squeeze(0)
        met = self.meta_model(x_meta)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, self.kernel_2 * self.output_D * self.output_D)
        H = self.feature_extractor_part2(H)

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)
        print(M.shape, A.shape, H.shape, n.shape)
        out = torch.cat([M, met])

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A.byte()

    def calculate_all(self, X, Y):
        Y = Y.float()
        Y_prob, Y_hat, A = self.forward(X)
        #Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))

        return neg_log_likelihood, error, Y_hat, A
