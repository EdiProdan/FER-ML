import json
import torch
from torch import nn
from sklearn.metrics import confusion_matrix
from time import time
from utils import *

DATA_DIR = 'datasets/cifar-10-batches-py/'
FILTERS_SAVE_DIR = 'out/cifar/1e-4/filters/'
EPOCHS_SAVE_DIR = 'out/cifar/1e-4/epochs/'

IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10

config = {'max_epochs': 30, 'batch_size': 50, 'weight_decay': 1e-4, 'lr': 1e-1}


class CovolutionalModel(nn.Module):
    def __init__(self, in_channels=3, conv1_width=16, conv2_width=32, fc1_width=256, class_count=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, padding=3, dtype=torch.float, bias=True)
        self.conv2 = nn.Conv2d(conv1_width, conv2_width, kernel_size=5, padding=3, dtype=torch.float, bias=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        fc1_in_features = 8 * 8 * conv2_width
        self.fc1 = nn.Linear(fc1_in_features, fc1_width, bias=True)
        self.fc2 = nn.Linear(in_features=256, out_features=128, bias=True)
        self.fc3_logits = nn.Linear(in_features=128, out_features=class_count, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.fc3_logits:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        self.fc3_logits.reset_parameters()

    def forward(self, x):
        h = self.conv1(x)
        h = torch.relu(h)
        h = self.maxpool(h)

        h = self.conv2(h)
        h = torch.relu(h)
        h = self.maxpool(h)

        h = torch.flatten(h, 1)

        h = self.fc1(h)
        h = torch.relu(h)
        h = self.fc2(h)
        h = torch.relu(h)
        logits = self.fc3_logits(h)
        return logits

    def train(self, train_x, train_y, valid_x, valid_y, config, criterion, optimizer, scheduler):
        start_time = time()
        total_time = start_time
        batch_size = config['batch_size']
        max_epochs = config['max_epochs']

        num_examples = train_x.shape[0]
        num_batches = num_examples // batch_size

        plot_data = {'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': [], 'lr': []}

        draw_conv_filters(0, 0, self.conv1.weight.detach().numpy(), FILTERS_SAVE_DIR)

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
            draw_conv_filters(epoch, 0, self.conv1.weight.detach().numpy(), FILTERS_SAVE_DIR)

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
            print(f"Total time elapsed: {(total_time - start_time)/60} minutes")

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

    train_x = np.ndarray((0, IMG_HEIGHT * IMG_WIDTH * NUM_CHANNELS), dtype=np.float32)
    train_y = []
    for i in range(1, 6):
        subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))

        train_x = np.vstack((train_x, subset['data']))
        train_y += subset['labels']
    train_x = train_x.reshape((-1, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH)).transpose(0, 2, 3, 1)
    train_y = np.array(train_y, dtype=np.int32)

    subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
    test_x = subset['data'].reshape((-1, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH)).transpose(0, 2, 3, 1).astype(np.float32)
    test_y = np.array(subset['labels'], dtype=np.int32)

    valid_size = 5000
    train_x, train_y = shuffle_data(train_x, train_y)
    valid_x = train_x[:valid_size, ...]
    valid_y = train_y[:valid_size, ...]
    train_x = train_x[valid_size:, ...]
    train_y = train_y[valid_size:, ...]
    data_mean = train_x.mean((0, 1, 2))
    data_std = train_x.std((0, 1, 2))

    train_x = (train_x - data_mean) / data_std
    valid_x = (valid_x - data_mean) / data_std
    test_x = (test_x - data_mean) / data_std

    train_x = train_x.transpose(0, 3, 1, 2)
    valid_x = valid_x.transpose(0, 3, 1, 2)
    test_x = test_x.transpose(0, 3, 1, 2)

    train_y = dense_to_one_hot(train_y, NUM_CLASSES)
    valid_y = dense_to_one_hot(valid_y, NUM_CLASSES)
    test_y = dense_to_one_hot(test_y, NUM_CLASSES)

    train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y, dtype=torch.float32)
    validate_x_tensor = torch.tensor(valid_x, dtype=torch.float32)
    validate_y_tensor = torch.tensor(valid_y, dtype=torch.float32)
    test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
    test_y_tensor = torch.tensor(test_y, dtype=torch.float32)

    conv = CovolutionalModel()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(conv.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85, last_epoch=-1)

    conv.train(train_x_tensor, train_y_tensor, validate_x_tensor,
               validate_y_tensor, config, criterion, optimizer, scheduler)

    loss_avg_test, acc_test, f1_test = conv.evaluate("Test", test_x_tensor, test_y_tensor, config, criterion)

    with open(f'{EPOCHS_SAVE_DIR}/test.json', 'w', encoding='utf-8') as json_file:
        json.dump({"test_loss": loss_avg_test, "test_acc": acc_test.item(), "f1_test": f1_test}, json_file, indent=4)

