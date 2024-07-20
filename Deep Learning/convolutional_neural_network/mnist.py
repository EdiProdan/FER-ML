import json

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn

from torchvision.datasets import MNIST
from time import time

from utils import dense_to_one_hot, plot_training_progress, draw_conv_filters_grey

DATA_DIR = 'datasets/MNIST/'
FILTERS_SAVE_DIR = 'out/MNIST/1e-1/filters/'
EPOCHS_SAVE_DIR = 'out/MNIST/1e-1/epochs/'

config = {'max_epochs': 8, 'batch_size': 50, 'weight_decay': 1e-1, 'lr': 1e-1}


class CovolutionalModel(nn.Module):
    def __init__(self, in_channels=1, conv1_width=16, conv2_width=32, fc1_width=512, class_count=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2 = nn.Conv2d(conv1_width, conv2_width, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        fc1_in_features = 7 * 7 * conv2_width
        self.fc1 = nn.Linear(fc1_in_features, fc1_width, bias=True)
        self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.fc_logits:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        self.fc_logits.reset_parameters()

    def forward(self, x):
        h = self.conv1(x)
        h = torch.relu(h)
        h = self.pool(h)

        h = self.conv2(h)
        h = torch.relu(h)
        h = self.pool(h)

        h = h.view(h.size(0), -1)

        h = self.fc1(h)
        h = torch.relu(h)
        logits = self.fc_logits(h)
        return logits

    def train(self, train_x, train_y, valid_x, valid_y, config, criterion, optimizer, scheduler):
        start_time = time()
        total_time = start_time
        batch_size = config['batch_size']
        max_epochs = config['max_epochs']

        num_examples = train_x.shape[0]
        num_batches = num_examples // batch_size

        plot_data = {'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': [], 'lr': []}

        draw_conv_filters_grey(0, 0, self.conv1.weight.detach().numpy(), FILTERS_SAVE_DIR)

        for epoch in range(1, max_epochs + 1):
            cnt_correct = 0
            permutation_idx = np.random.permutation(num_examples)
            train_x = train_x[permutation_idx]
            train_y = train_y[permutation_idx]

            loss_avg = []
            for i in range(num_batches):
                batch_x = train_x[i * batch_size:(i + 1) * batch_size, :]
                batch_y = train_y[i * batch_size:(i + 1) * batch_size, :]
                logits = self.forward(batch_x)
                loss = criterion(logits, torch.argmax(batch_y, dim=1))
                loss_avg.append(loss)

                yp = torch.argmax(logits, 1)
                yt = torch.argmax(batch_y, 1)
                cnt_correct += (yp == yt).sum()

                loss.backward()
                optimizer.step()

                if i % 5 == 0:
                    print("epoch %d, step %d/%d, batch loss = %.2f" %
                          (epoch, i * batch_size, num_examples, loss))
                if i > 0 and i % 50 == 0:
                    print("Train accuracy = %.2f" %
                          (cnt_correct / ((i + 1) * batch_size) * 100))
                optimizer.zero_grad()

            plot_data['lr'] += [scheduler.get_last_lr()]
            scheduler.step()

            draw_conv_filters_grey(epoch, 0, self.conv1.weight.detach().numpy(), FILTERS_SAVE_DIR)

            loss_avg_valid, acc_valid, f1_valid = self.evaluate("Validation", valid_x,
                                                                valid_y,
                                                                config, criterion)
            loss_avg_train, acc_train, f1_train = self.evaluate("Train", train_x,
                                                                train_y,
                                                                config, criterion)

            plot_data['train_loss'] += [loss_avg_train]
            plot_data['valid_loss'] += [loss_avg_valid]
            plot_data['train_acc'] += [acc_train]
            plot_data['valid_acc'] += [acc_valid]

            print(f"Epoch {epoch} took {time() - total_time} seconds")
            total_time = time()
            print(f"Total time elapsed: {total_time - start_time} seconds")

            with open(f'{EPOCHS_SAVE_DIR}/epoch_{epoch}.json', 'w', encoding='utf-8') as json_file:
                json.dump({"valid_loss": loss_avg_valid, "train_loss": loss_avg_train, "valid_acc": acc_valid.item(),
                           "train_acc": acc_train.item(), "f1_validation": f1_valid, "f1_train": f1_train}, json_file,
                          indent=4)

        plot_training_progress(f'{EPOCHS_SAVE_DIR}', plot_data)

    def evaluate(self, name, x_valid, y_valid, config, criterion):
        print(f"{name} evaluation: ")

        batch_size = config['batch_size']

        num_examples = x_valid.shape[0]
        assert num_examples % batch_size == 0
        num_batches = num_examples // batch_size

        cnt_correct = 0
        loss_avg = 0
        for i in range(num_batches):
            batch_x = x_valid[i * batch_size:(i + 1) * batch_size, :]
            batch_y = y_valid[i * batch_size:(i + 1) * batch_size, :]
            logits = self.forward(batch_x)
            yp = torch.argmax(logits, 1)
            yt = torch.argmax(batch_y, 1)
            cnt_correct += (yp == yt).sum()

            loss_val = criterion(logits, yt)
            loss_avg += loss_val.item()

        valid_acc = cnt_correct / num_examples * 100
        loss_avg /= num_batches

        cm = confusion_matrix(yt, yp)
        sums = [np.sum(cm, axis=y) for y in [None, 0, 1]]
        sums[0] = np.maximum(1, sums[0])
        for i in range(1, len(sums)):
            sums[i][sums[i] == 0] = 1

        precision = np.mean(np.diag(cm) / sums[1])
        recall = np.mean(np.diag(cm) / sums[2])
        try:
            f1 = (2 * precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1 = 0
        print("Accuracy = %.2f" % valid_acc)
        print("Average Loss = %.2f\n" % loss_avg)

        return loss_avg, valid_acc, f1


if __name__ == '__main__':
    # np.random.seed(100)
    np.random.seed(int(time() * 1e6) % 2 ** 31)
    ds_train, ds_test = MNIST(DATA_DIR, train=True, download=True), MNIST(DATA_DIR, train=False)

    train_x = ds_train.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float32) / 255
    train_y = ds_train.targets.numpy()
    train_x, valid_x = train_x[:55000], train_x[55000:]
    train_y, valid_y = train_y[:55000], train_y[55000:]
    test_x = ds_test.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float32) / 255
    test_y = ds_test.targets.numpy()
    train_mean = train_x.mean()
    train_x, valid_x, test_x = (x - train_mean for x in (train_x, valid_x, test_x))
    train_y, valid_y, test_y = (dense_to_one_hot(y, 10) for y in (train_y, valid_y, test_y))

    train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y, dtype=torch.float32)
    valid_x_tensor = torch.tensor(valid_x, dtype=torch.float32)
    valid_y_tensor = torch.tensor(valid_y, dtype=torch.float32)
    test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
    test_y_tensor = torch.tensor(test_y, dtype=torch.float32)

    conv = CovolutionalModel()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(conv.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 5, 7], gamma=0.1)

    conv.train(train_x_tensor, train_y_tensor, valid_x_tensor, valid_y_tensor, config, criterion, optimizer, scheduler)
    loss_avg_test, acc_test, f1_test = conv.evaluate("Test", test_x_tensor, test_y_tensor, config, criterion)

    with open(f'{EPOCHS_SAVE_DIR}/test.json', 'w', encoding='utf-8') as json_file:
        json.dump({"test_loss": loss_avg_test, "test_acc": acc_test.item(), "f1_test": f1_test}, json_file, indent=4)
