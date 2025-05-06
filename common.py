from collections import Counter

import torch.nn as nn
from tqdm import tqdm

EMBED_DIM = 16
NUM_CLASS = 2


def word_tokenize(text):
    return text.split()


def build_vocab(texts, min_freq=1):
    counter = Counter()
    for text in tqdm(texts, desc="Building vocabulary"):
        tokens = word_tokenize(text)
        counter.update(tokens)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    print(f"Vocabulary size: {len(vocab)}")
    return vocab


def encode(text, vocab):
    return [vocab.get(token, vocab['<UNK>']) for token in word_tokenize(text)]


def pad_sequence(seq, max_length, pad_value=0):
    if len(seq) < max_length:
        return seq + [pad_value] * (max_length - len(seq))
    else:
        return seq[:max_length]


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)
        return self.fc(pooled)
