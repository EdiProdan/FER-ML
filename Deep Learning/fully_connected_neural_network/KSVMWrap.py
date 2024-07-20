from matplotlib import pyplot as plt
from sklearn.svm import SVC

import data
import numpy as np


class KSVMWrap:

    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.model = SVC(C=param_svm_c, gamma=param_svm_gamma, kernel='rbf').fit(X, Y_)

    def predict(self, X):
        return self.model.predict(X)

    def get_scores(self, X):
        return self.model.decision_function(X)

    def support(self):
        return self.model.support_


if __name__ == '__main__':

    _param_svm_c = 1
    _param_svm_gamma = 'auto'

    np.random.seed(100)

    X, Y_ = data.sample_gmm_2d(4, 2, 40)
    Yoh_ = data.class_to_onehot(Y_)

    model = KSVMWrap(X, Y_)

    Y = model.predict(X)

    boxx = (np.min(X, axis=0), np.max(X, axis=0))

    data.graph_surface(model.predict, boxx, 0.5, 1024, 1024)
    data.graph_data(X, Y_, model.predict(X), model.support())

    accuracy, pr, M = data.eval_perf_multi(model.predict(X), Y_)
    precision, recall = pr
    precision = np.mean(precision)
    recall = np.mean(recall)

    with open(f"lab/lab1/results/svm/4_2_40.txt", "w") as f:
        f.write(f'Accuracy: {accuracy:.4f}\n')
        f.write(f'Precision: {precision:.4f}\n')
        f.write(f'Recall: {recall:.4f}\n')

    plt.savefig(f"lab/lab1/plots/svm/4_2_40.png")

