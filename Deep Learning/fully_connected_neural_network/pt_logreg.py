import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

import data


class PTLogreg(nn.Module):

    def __init__(self, D, C):
        super().__init__()
        self.W = nn.Parameter(torch.randn(D, C))
        self.b = nn.Parameter(torch.zeros(C))

    def forward(self, X) -> torch.Tensor:
        scores = X.mm(self.W) + self.b
        return torch.softmax(scores, dim=1)

    def get_loss(self, X, Yoh_, param_lambda) -> torch.Tensor:
        probs = self.forward(X)
        return -torch.mean(torch.sum(Yoh_ * torch.log(probs), dim=1)) + param_lambda * torch.linalg.norm(self.W)


def train(model, X, Yoh_, param_niter, param_delta, param_lambda=0.0):

    optimizer = torch.optim.SGD(model.parameters(), lr=param_delta, weight_decay=param_lambda)

    for i in range(1, param_niter + 1):
        loss = model.get_loss(X, Yoh_, param_lambda) + param_lambda * torch.linalg.norm(model.W)

        if i % 100 == 0:
            print(f'iteration {i}: loss {loss.item()}')

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def logreg_eval(model: PTLogreg, X: np.array) -> np.array:
    X = torch.Tensor(X)
    return model.forward(X).detach().numpy()


def logreg_decfun(ptlr: PTLogreg) -> callable:
    def eval(X: np.array) -> np.array:
        return np.argmax(logreg_eval(ptlr, X), axis=1)
    return eval


if __name__ == "__main__":
    np.random.seed(100)

    X, Y_ = data.sample_gmm_2d(6, 3, 10)
    Yoh_ = data.class_to_onehot(Y_)

    param_niter, param_delta = 10000, 0.01
    param_lambda_list = [0.001, 0.01, 0.1, 0, 0.5, 1]
    for param_lambda in param_lambda_list:
        X = torch.Tensor(X)
        Yoh_ = torch.Tensor(Yoh_)

        model = PTLogreg(X.shape[1], Yoh_.shape[1])

        train(model, X, Yoh_, param_niter, param_delta, param_lambda)
        print(f"Training with lambda={param_lambda}")

        X = X.detach().numpy()

        probs = logreg_eval(model, X)

        Y = np.argmax(probs, axis=1)

        accuracy, pr, M = data.eval_perf_multi(Y, Y_)

        boxx = (np.min(X, axis=0), np.max(X, axis=0))

        decfun = logreg_decfun(model)

        plt.clf()

        data.graph_surface(decfun, boxx, offset=0.5)
        data.graph_data(X, Y_, Y, special=[])
        plt.title(f"lambda={param_lambda}")
        # plt.savefig(f"lab/lab1/plots/pt_logreg/lambda_{str(param_lambda)}.png")
