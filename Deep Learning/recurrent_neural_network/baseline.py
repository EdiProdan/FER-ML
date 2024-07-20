import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn

from lab.lab3.src.data_preprocessing import NLPDataset, pad_collate_fn
from utils import metrics, write_metrics

EPOCHS = 5


class BaselineModel(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()

        self.fc1 = nn.Linear(300, 150)
        self.fc2 = nn.Linear(150, 150)
        self.fc3 = nn.Linear(150, 1)
        self.relu = nn.ReLU()

        self.embedding = embedding_matrix

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x, lengths):
        h = self.embedding(x)
        packed_input = pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = pad_packed_sequence(packed_input, batch_first=True)
        h = torch.sum(packed_output, dim=1) / lengths.unsqueeze(1).float()
        h = self.fc1(h)
        h = self.relu(h)
        h = self.fc2(h)
        h = self.relu(h)
        return self.fc3(h)


def train(model, data, optimizer, criterion):
    model.train()
    for batch_num, batch in enumerate(data):
        texts, labels, lengths = batch
        model.zero_grad()
        logits = model.forward(texts, lengths).squeeze(-1)
        loss = criterion(logits, labels.float())
        loss.backward()
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
        write_metrics(acc, f1, f"../results/task_2/{path_str}.txt", epoch)
    else:
        write_metrics(acc, f1, f"../results/task_2/{path_str}.txt")


if __name__ == '__main__':
    seed = 842142

    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset = NLPDataset(path="../data/sst_train_raw.csv")
    embedding_matrix = train_dataset.text_vocab.generate_embedding_matrix("../data/sst_glove_6B_300d.txt")
    test_dataset = NLPDataset("../data/sst_test_raw.csv", train_dataset.text_vocab, train_dataset.label_vocab)
    valid_dataset = NLPDataset("../data/sst_valid_raw.csv", train_dataset.text_vocab, train_dataset.label_vocab)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True,
                                                   batch_size=10, collate_fn=pad_collate_fn)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=True,
                                                  batch_size=32, collate_fn=pad_collate_fn)
    valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, shuffle=True,
                                                   batch_size=32, collate_fn=pad_collate_fn)

    baseline_model = BaselineModel(embedding_matrix)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=1e-4)

    for epoch in range(1, EPOCHS + 1):
        train(baseline_model, train_dataloader, optimizer, criterion)
        evaluate(baseline_model, valid_dataloader, criterion, "Valid", epoch)

    evaluate(baseline_model, test_dataloader, criterion, "Test")
