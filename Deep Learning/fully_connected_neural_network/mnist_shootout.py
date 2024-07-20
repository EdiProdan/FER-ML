import torch
import torchvision
from matplotlib import pyplot as plt

from pt_deep import PTDeep
from pt_deep import train, deep_eval
import data
import numpy as np


if __name__ == "__main__":
    dataset_root = 'lab/lab1/tmp/mnist'
    mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets
    x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

    y_train_oh = data.class_to_onehot(y_train)
    y_test_oh = data.class_to_onehot(y_test)

    x_train = torch.clone(x_train).detach().reshape(-1, 784)
    x_test = torch.clone(x_test).detach().reshape(-1, 784)

    D = x_train.shape[1]
    C = y_train_oh.shape[1]
    N = x_train.shape[0]

    configurations = [[784, 10], [784, 100, 10]]

    # plt.imshow(x_train[9].reshape(28, 28))
    # plt.show()

    for config in configurations:
        hidden_layers = config[1:-1]

        model = PTDeep(D, C, hidden_layers, torch.nn.ReLU())

        train(model, x_train, torch.tensor(y_train_oh), 2000, 0.1, 0.001)

        for i in range(20):
            with open(f"lab/lab1/results/mnist/mnist_{i}.txt", "w") as f:
                f.write(f"{model.weights[0][i].detach().numpy()}")

        Y = deep_eval(model, x_test.detach().numpy())
        accuracy, pr, M = data.eval_perf_multi(Y, y_test)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {np.mean(pr[0])}")
        print(f"Recall: {np.mean(pr[1])}")
