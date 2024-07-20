import numpy as np
import matplotlib.pyplot as plt
# import softmax from scipy.special
# from scipy.special import softmax
import math
import data


def relu(x: np.array) -> np.array:
    return np.maximum(0, x)


def softmax(x: np.array) -> np.array:
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def fcann2_train(X: np.array, Y_: np.array, hidden_dim: int, param_niter: int,
                 param_delta: float, param_lambda: float) -> (np.array, float, np.array, float):

    Yoh = data.class_to_onehot(Y_)
    N, D = X.shape
    C = Yoh.shape[1]

    W1 = np.random.randn(hidden_dim, D)  # H x D
    b1 = np.zeros(hidden_dim)  # H x 1
    W2 = np.random.randn(C, hidden_dim)  # P x H
    b2 = np.zeros(C)  # P x 1

    for i in range(param_niter):

        s1 = np.dot(X, W1.T) + b1
        h = relu(s1)
        s2 = np.dot(h, W2.T) + b2
        probs = softmax(s2)

        loss = -np.mean(np.log(np.sum(probs * Yoh, axis=1))) + param_lambda * (np.sum(W1 ** 2) + np.sum(W2 ** 2))

        if i % 100 == 0:
            print("iteration {}: plot_loss {}".format(i, loss))

        Gs2 = probs - Yoh  # N x C
        grad_W2 = np.dot(Gs2.T, h) / N  # C x H
        grad_b2 = np.sum(Gs2, axis=0) / N  # C x 1
        Gh1 = np.dot(Gs2, W2) * np.where(s1 > 0, 1, 0)  # N x H
        Gs1 = Gh1 * np.where(s1 > 0, 1, 0)  # N x H

        grad_W1 = np.dot(Gs1.T, X) / N  # H x D

        grad_b1 = np.sum(Gs1, axis=0) / N  # H x 1

        W1 -= param_delta * grad_W1
        b1 -= param_delta * grad_b1
        W2 -= param_delta * grad_W2
        b2 -= param_delta * grad_b2

    return W1, b1, W2, b2


def fcann2_classify(X: np.array, W1: np.array, b1: float, W2: np.array, b2: float) -> np.array:
    s1 = np.dot(X, W1.T) + b1
    h = relu(s1)
    s2 = np.dot(h, W2.T) + b2

    return softmax(s2)


def fcann2_decfun(W1: np.array,b1: float, W2: np.array,b2: float) -> callable:
    def classify(X: np.array) -> np.array:
        return np.argmax(fcann2_classify(X, W1, b1, W2, b2), axis=1)
    return classify


if __name__ == "__main__":
    np.random.seed(100)

    X, Y_ = data.sample_gmm_2d(6, 2, 10)

    W1, b1, W2, b2 = fcann2_train(X, Y_, 5, int(1e5), 0.05, 1e-3)

    probs = fcann2_classify(X, W1, b1, W2, b2)

    Y = np.argmax(probs, axis=1)

    decfun = fcann2_decfun(W1, b1, W2, b2)

    accuracy, pr, M = data.eval_perf_multi(Y, Y_)

    print("Accuracy: ", accuracy)
    print("Precision: ", np.mean(pr[0]))
    print("Recall: ", np.mean(pr[1]))

    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    data.graph_data(X, Y_, Y, special=[])

    plt.savefig("lab/lab1/plots/fcann2.png")
    # plt.show()
