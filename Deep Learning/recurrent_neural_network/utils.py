from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def metrics(y_pred, y_true):
    acc = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred, normalize='true').diagonal()
    f1 = f1_score(y_true, y_pred)

    return acc, conf_matrix, f1


def write_metrics(acc, f1, path, epoch=None):
    with open(path, 'a') as file:
        if epoch:
            file.write(f"Epoch: {epoch}\n")
            file.write(f"\tAccuracy: {acc}\n")
            file.write(f"\tF1: {f1}\n")
            file.write("\n")
        else:
            file.write(f"Test Accuracy: {acc}\n")
            file.write(f"Test F1: {f1}\n")
            file.write("\n")


parameters_list = [
    # [Vocabulary Size, Train Batch Size, Test Batch Size, Dropout, Number of Layers, Hidden Size]
    [2500, 5, 16, 0, 1, 50],
    [2500, 10, 32, 0.25, 2, 100],
    [2500, 30, 64, 0.5, 3, 150],

    [7000, 5, 16, 0.25, 1, 100],
    [7000, 10, 32, 0.5, 2, 150],
    [7000, 30, 64, 0, 3, 50],

    [14806, 5, 16, 0.5, 1, 150],
    [14806, 10, 32, 0, 2, 50],
    [14806, 30, 64, 0.25, 3, 100],

    [2500, 5, 32, 0.5, 1, 100],
    [7000, 10, 16, 0, 2, 150],
    [14806, 30, 32, 0.25, 3, 50],

    [2500, 10, 64, 0.25, 1, 150],
    [7000, 5, 32, 0.5, 2, 50],
    [14806, 10, 16, 0, 3, 100]
]
