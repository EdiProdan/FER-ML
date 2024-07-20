import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

import data


class PTDeep(nn.Module):
    def __init__(self, D, C, hidden_layers, activ_f):
        """Arguments:
            - D: dimensions of each datapoint # 2
            - C: number of classes # 2
            - hidden_layers: number of neurons in each hidden layer  # [5]
        """
        super().__init__()
        if hidden_layers:
            Ws = [nn.Parameter(torch.randn(D, hidden_layers[0]))]
            bs = [nn.Parameter(torch.zeros(hidden_layers[0]))]
            for i in range(1, len(hidden_layers)):
                Ws.append(nn.Parameter(torch.randn(hidden_layers[i-1], hidden_layers[i])))
                bs.append(nn.Parameter(torch.zeros(hidden_layers[i])))

            Ws.append(nn.Parameter(torch.randn(hidden_layers[-1], C)))
            bs.append(nn.Parameter(torch.zeros(C)))
        else:
            Ws = [nn.Parameter(torch.randn(D, C))]
            bs = [nn.Parameter(torch.zeros(C))]

        self.weights = nn.ParameterList(Ws)
        self.biases = nn.ParameterList(bs)

        self.activ_f = activ_f

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        for i in range(0, len(self.weights) - 1):
            s = X.mm(self.weights[i]) + self.biases[i]
            X = self.activ_f(s)
        s = X.mm(self.weights[-1]) + self.biases[-1]
        return torch.softmax(s, dim=1)

    def get_loss(self, X: torch.Tensor, Yoh_, param_lambda) -> torch.Tensor:
        probs = self.forward(X)

        reg_term = 0
        for weight in self.weights:
            reg_term += torch.sum(weight ** 2)
        return -torch.mean(torch.sum(Yoh_ * torch.log(probs + 1e-13), dim=1)) + param_lambda * reg_term


def train(model, X: torch.Tensor, Yoh_, param_niter, param_delta, param_lambda=0.0):
    optimizer = torch.optim.SGD(model.parameters(), lr=param_delta, weight_decay=param_lambda)

    for i in range(1, param_niter + 1):
        loss = model.get_loss(X, Yoh_, param_lambda)

        if i % 100 == 0:
            print(f'iteration {i}: loss {loss.item()}')

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def get_count(model: PTDeep) -> int:
    return sum(p.numel() for p in model.parameters())


def deep_eval(model: PTDeep, X: np.array) -> np.array:
    X = torch.Tensor(X)
    return model.forward(X).detach().numpy()


def deep_decfun(model: PTDeep) -> callable:
    def eval(X: np.array) -> np.array:
        return np.argmax(deep_eval(model, X), axis=1)
    return eval


if __name__ == "__main__":
    np.random.seed(100)

    configurations = [[2, 5, 2], [2, 10, 2], [2, 10, 10, 2]]

    for i, configuration in enumerate(configurations):
        X, Y_ = data.sample_gmm_2d(6, 2, 10)
        Yoh_ = data.class_to_onehot(Y_)

        X = torch.Tensor(X)
        Yoh_ = torch.Tensor(Yoh_)

        D = configuration[0]  # dimensionality == 2
        C = configuration[-1]  # number of classes == 3
        hidden_layers = configuration[1:-1]

        activ_f = nn.ReLU()
        # activ_f = nn.Sigmoid()
        # activ_f = nn.tanh()

        ptd = PTDeep(D, C, hidden_layers, activ_f)

        num_params = get_count(ptd)
        print(f"Number of parameters: {num_params}")

        train(ptd, X, Yoh_, 10000, 0.1, 0.001)

        X = X.detach().numpy()

        probs = deep_eval(ptd, X)

        Y = np.argmax(probs, axis=1)

        boxx = (np.min(X, axis=0), np.max(X, axis=0))

        decfun = deep_decfun(ptd)

        plt.clf()

        accuracy, pr, M = data.eval_perf_multi(Y, Y_)
        precision, recall = pr
        precision = np.mean(precision)
        recall = np.mean(recall)

        with open(f"lab/lab1/results/pt_deep/6_2_10_sigm.txt", "w") as f:
            f.write(f'Accuracy: {accuracy:.4f}\n')
            f.write(f'Precision: {precision:.4f}\n')
            f.write(f'Recall: {recall:.4f}\n')

        data.graph_surface(decfun, boxx, offset=0.5)
        data.graph_data(X, Y_, Y, special=[])
        plt.title(f"Configuration: {configuration}")
        # plt.savefig(f"lab/lab1/plots/pt_deep/sample_gmm(4,2,40)_{i}_sigm.png")
