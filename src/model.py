import torch
import torch.nn.functional as F
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 512 # 512 node fully connected layer
        self.D = 128 # 128 node attention layer
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 36, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(36, 48, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(48 * 30 * 30, self.L),
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

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 48 * 30 * 30)
        H = self.feature_extractor_part2(H)

        A = self.attention(H) # NxK
        A = torch.transpose(A, 1, 0) # KxN
        A = F.softmax(A, dim=1) # softmax over N

        M = torch.mm(A, H)

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A.byte()

  def calculate_classification_error(self, X, Y):
      Y = Y.float()
      _, Y_hat, _ = self.forward(X)
      error = 1. - Y_hat.eq(Y).cpu().float().mean().data

      return error, Y_hat

  def calculate_objective(self, X, Y):
      Y = Y.float()
      Y_prob, _, A = self.forward(X)
      Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
      neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))

      return neg_log_likelihood, A