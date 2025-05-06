import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from common import TextClassifier, pad_sequence, encode, EMBED_DIM, NUM_CLASS


class MyDataset(Dataset):
    def __init__(self, dataframe, vocab, max_length):
        self.texts = dataframe['text'].tolist()
        self.vocab = vocab
        self.max_length = max_length
        self.encoded_texts = [
            pad_sequence(encode(text, self.vocab), self.max_length)
            for text in self.texts
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded_texts[idx], dtype=torch.long)


def main():
    test_df = pd.read_csv("test_split.csv")

    checkpoint = torch.load("model.pth")
    vocab = checkpoint['vocab']
    max_length = checkpoint['max_length']

    model = TextClassifier(len(vocab), EMBED_DIM, NUM_CLASS)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_dataset = MyDataset(test_df, vocab, max_length)
    test_loader = DataLoader(test_dataset, batch_size=64)

    all_predictions = []
    with torch.no_grad():
        for texts in tqdm(test_loader):
            outputs = model(texts)
            preds = outputs.argmax(dim=1).tolist()
            preds = [pred + 1 for pred in preds]
            all_predictions.extend(preds)

    test_df['predictions'] = all_predictions
    test_df.to_csv('predictions.csv', index=False)

    correct = sum([1 for i, pred in enumerate(all_predictions) if pred == test_df['label'].iloc[i]])
    accuracy = correct / len(test_df)

    with open('metrics.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy:.4f}\n')


if __name__ == "__main__":
    main()
