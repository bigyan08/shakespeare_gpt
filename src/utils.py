import torch
from sklearn.model_selection import train_test_split

test_size = 0.2
seed = 42
def load_data(path, test_size=0.2, seed=42):
    text = open(path, "r", encoding="utf-8").read()
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    # train/test split
    train_text, val_text = train_test_split(
        list(text), test_size=test_size, random_state=seed
    )
    train_text = "".join(train_text)
    val_text = "".join(val_text)

    return train_text, val_text, vocab, stoi, itos, encode, decode


class CharDataset(torch.utils.data.Dataset):
    def __init__(self, data, block_size, encode):
        self.data = data
        self.block_size = block_size
        self.encode = encode

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx+self.block_size+1]
        x = torch.tensor(self.encode(chunk[:-1]), dtype=torch.long)
        y = torch.tensor(self.encode(chunk[1:]), dtype=torch.long)
        return x, y
