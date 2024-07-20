import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn

from lab.lab3.src.data_preprocessing import NLPDataset, pad_collate_fn
from utils import metrics, write_metrics, parameters_list

EPOCHS = 5
PATH = "../results/task_4/"


class VanillaRNNModel(nn.Module):
    def __init__(self, embedding_matrix, rnn=nn.RNN, hidden_size=150, num_layers=1,
                 bidirectional=False, dropout=0, gradient_clip=0.25, idx=-1):
        super().__init__()

        self.idx = idx

        self.rnn1 = rnn(300, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                           bidirectional=bidirectional, batch_first=True)
        self.rnn2 = rnn(hidden_size * (2 if bidirectional else 1), hidden_size=hidden_size, num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional, batch_first=True)

        self.fc1 = nn.Linear(hidden_size * (2 if bidirectional else 1), hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.loss = nn.BCEWithLogitsLoss()

        self.embedding = embedding_matrix
        self.clip = gradient_clip

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_output, _ = self.rnn1(packed_embedded)
        packed_output, _ = self.rnn2(packed_output)

        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(output.size(0), 1, output.size(2))
        h = output.gather(1, idx).squeeze(1)

        h = self.fc1(h)
        h = self.relu(h)
        return self.fc2(h)


def train(model, data, optimizer, criterion):
    model.train()
    for batch_num, batch in enumerate(data):
        texts, labels, lengths = batch
        model.zero_grad()
        logits = model.forward(texts, lengths).squeeze(-1)
        loss = criterion(logits, labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), model.clip)
        optimizer.step()


def evaluate(model, data, criterion, eval_str, epoch=None):
    print(f"Evaluating {eval_str}")
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_num, batch in enumerate(data):
            texts, labels, lengths = batch
            logits = model.forward(texts, lengths).squeeze(-1)
            loss = criterion(logits, labels.float())
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).int()
            all_preds.append(preds)
            all_labels.append(labels)
    avg_loss = total_loss / (batch_num + 1)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc, conf_matrix, f1 = metrics(all_preds, all_labels)
    if epoch:
        print(f"Epoch: {epoch}")
    print(f"Loss: {avg_loss}")
    print(f"Accuracy: {acc}")
    print(f"F1: {f1}")
    print("\n")
    path_str = str(seed)
    if epoch:
        write_metrics(acc, f1, f"../results/task_4/models/gru.txt", epoch)
    else:
        write_metrics(acc, f1, f"../results/task_4/models/gru.txt")


if __name__ == '__main__':
    seed = 7052020

    np.random.seed(seed)
    torch.manual_seed(seed)

    # best = [7000, 5, 16, 0.25, 1, 100]
    # seed_list = [42, 871, 1024, 842141, 7052020]
    # max_size, train_batch_size, test_batch_size, dropout, num_layers, hidden_size = best
    # print(max_size, train_batch_size, test_batch_size, dropout, num_layers, hidden_size)

    rnn_list = [nn.LSTM, nn.GRU]

    for rnn in rnn_list:

        train_dataset = NLPDataset(path="../data/sst_train_raw.csv")
        embedding_matrix = train_dataset.text_vocab.generate_embedding_matrix("../data/sst_glove_6B_300d.txt")
        test_dataset = NLPDataset("../data/sst_test_raw.csv", train_dataset.text_vocab, train_dataset.label_vocab)
        valid_dataset = NLPDataset("../data/sst_valid_raw.csv", train_dataset.text_vocab, train_dataset.label_vocab)

        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=10,
                                                       collate_fn=pad_collate_fn)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=True, batch_size=32,
                                                      collate_fn=pad_collate_fn)
        valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, shuffle=True, batch_size=32,
                                                       collate_fn=pad_collate_fn)

        model = VanillaRNNModel(embedding_matrix, num_layers=2, rnn=rnn)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(1, EPOCHS + 1):
            train(model, train_dataloader, optimizer, criterion)
            evaluate(model, valid_dataloader, criterion, "Valid", epoch)

        evaluate(model, test_dataloader, criterion, "Test")
