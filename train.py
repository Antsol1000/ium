import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from common import build_vocab, TextClassifier, pad_sequence, encode, EMBED_DIM, NUM_CLASS

# Configuration
train_file = 'path/to/train.csv'
dev_file = 'path/to/dev.csv'
batch_size = 64
max_length = 250
learning_rate = 1e-3

class MyDataset(Dataset):
    def __init__(self, dataframe, vocab, max_length):
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['label'].tolist()
        self.max_length = max_length
        self.vocab = vocab

        self.encoded_texts = [
            pad_sequence(encode(text, self.vocab), self.max_length)
            for text in self.texts
        ]
        self.labels = [int(label) - 1 for label in self.labels]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_tensor = torch.tensor(self.encoded_texts[idx], dtype=torch.long)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return text_tensor, label_tensor

def main(epochs):
    print("Loading data...")
    train_df = pd.read_csv(train_file)
    dev_df = pd.read_csv(dev_file)

    print("Building vocabulary...")
    vocab = build_vocab(train_df.text.tolist(), min_freq=2)

    print("Preparing datasets...")
    train_dataset = MyDataset(train_df, vocab=vocab, max_length=max_length)
    dev_dataset = MyDataset(dev_df, vocab=vocab, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    model = TextClassifier(len(vocab), EMBED_DIM, NUM_CLASS)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for texts, labels in tqdm(train_loader, desc=f"Training epoch {epoch + 1}"):
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        scheduler.step()
        print(f"Current learning rate: {scheduler.get_last_lr()[0]}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for texts, labels in tqdm(dev_loader, desc="Validation"):
                outputs = model(texts)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")

    print("Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'max_length': max_length,
    }, "model.pth")
    print("Model saved to model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a text classification model.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    args = parser.parse_args()
    
    main(args.epochs)

