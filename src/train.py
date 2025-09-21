from src.utils import load_data, CharDataset
import src.config as cfg
import torch
from src.model import Transformer
from torch.utils.data import DataLoader

def train():
    train_text, val_text, vocab, stoi, itos, encode, decode = load_data(cfg.DATA_PATH)
    cfg.VOCAB_SIZE = len(vocab)

    train_dataset = CharDataset(train_text, cfg.BLOCK_SIZE, encode)
    val_dataset = CharDataset(val_text, cfg.BLOCK_SIZE, encode)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE)

    model = Transformer().to(cfg.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(cfg.EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            print(f'{loss}\n')
            opt.zero_grad()
            loss.backward()
            opt.step()

        torch.save(model.state_dict(),'../models/mini_gpt.pth')
        # validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for x, y in val_loader:
                x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
                logits = model(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

if __name__=='__main__':
    train()