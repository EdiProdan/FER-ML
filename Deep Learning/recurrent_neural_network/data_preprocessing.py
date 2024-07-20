from dataclasses import dataclass
from typing import List, Tuple

from torch import Tensor
from torch.utils.data import Dataset
import csv

import torch


def pad_collate_fn(batch) -> Tuple[Tensor, Tensor, Tensor]:
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    max_length = max(lengths)

    padded_texts = torch.zeros(len(texts), max_length, dtype=torch.long)
    for i, text in enumerate(texts):
        padded_texts[i, :len(text)] = text

    return padded_texts, torch.tensor(labels), lengths


@dataclass
class Instance:
    text: List[str]
    label: str

    def __iter__(self):
        return iter((self.text, self.label))


def read_csv(path: str) -> List[Instance]:
    instances = []
    with open(path, newline='') as csvfile:
        csv_instances = csv.reader(csvfile, delimiter=',')
        for text, label in csv_instances:
            instances.append(Instance(text.split(), label))
    return instances


class Vocab:

    def __init__(self, freq: dict, max_size: int = -1, min_freq: int = 0, label=False):
        if not label:
            self.itos, self.stoi = ["<UNK>", "<PAD>"], {"<UNK>": 0, "<PAD>": 1}
        else:
            self.itos, self.stoi = [], {}
        for word in freq.keys():
            if freq[word] > min_freq:
                self.itos.append(word)
                self.stoi[word] = len(self.itos) - 1
            if len(self.itos) == max_size:
                break

    def encode(self, tokens: List[str]) -> Tensor:
        if isinstance(tokens, str):
            return torch.tensor(self.stoi.get(tokens, 0))

        return torch.tensor([self.stoi.get(token, 0) for token in tokens])

    def generate_embedding_matrix(self, file) -> Tensor:
        embedding_dim = 300
        data = torch.randn(len(self.itos), embedding_dim)
        data[0] = torch.zeros(embedding_dim)

        if file:
            with open(file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    word = parts.pop(0)
                    vector = list(map(float, parts))
                    if word in self.stoi and len(vector) == embedding_dim:
                        data[self.stoi[word]] = torch.tensor(vector)

            return torch.nn.Embedding.from_pretrained(data, padding_idx=0, freeze=True)

        return torch.nn.Embedding.from_pretrained(data, padding_idx=0, freeze=False)


class NLPDataset(Dataset):
    def __init__(self, path: str = None, text_vocab: Vocab = None, label_vocab: Vocab = None, max_size=-1, min_freq=0):
        self.max_size = max_size
        self.min_freq = min_freq
        self.instances = read_csv(path)
        self.text_freq, self.label_freq = self.__get_frequencies()
        if text_vocab is None:
            self.text_vocab = Vocab(self.text_freq, self.max_size, self.min_freq)
        else:
            self.text_vocab = text_vocab
        if label_vocab is None:
            self.label_vocab = Vocab(self.label_freq, self.max_size, self.min_freq, label=True)
        else:
            self.label_vocab = label_vocab

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        text, label = self.instances[index]
        return self.text_vocab.encode(text), self.label_vocab.encode(label)

    def __len__(self) -> int:
        return len(self.instances)

    def __get_frequencies(self) -> Tuple[dict, dict]:
        text_freq = {}
        label_freq = {}
        for instance in self.instances:
            for word in instance.text:
                if word in text_freq:
                    text_freq[word] += 1
                else:
                    text_freq[word] = 1
            if instance.label in label_freq:
                label_freq[instance.label] += 1
            else:
                label_freq[instance.label] = 1

        text_freq = dict(sorted(text_freq.items(), key=lambda item: item[1], reverse=True))
        label_freq = dict(sorted(label_freq.items(), key=lambda item: item[1], reverse=True))

        return text_freq, label_freq
