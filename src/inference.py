import torch
import src.config as cfg
from src.model import Transformer
from src.utils import load_data

def generate_text(start, length):
    _, _, vocab, stoi, itos, encode, decode = load_data(cfg.DATA_PATH)
    cfg.VOCAB_SIZE = len(vocab)

    model = Transformer().to(cfg.DEVICE)
    model.load_state_dict(torch.load("./models/mini_gpt.pth", map_location=cfg.DEVICE))
    model.eval()

    idx = torch.tensor([encode(start)], dtype=torch.long).to(cfg.DEVICE)

    for _ in range(length):
        '''
        if seq_len is > block size then truncate it to the last 128(blocksize) tokens.
        Last tokens instead of first 128 tokens because recency matters, the most recent tokens 
        has high priority in deciding the next tokens.
        '''
        if idx.size(1) > cfg.BLOCK_SIZE:
            idx_cond = idx[:,-cfg.BLOCK_SIZE:]
        else:
            idx_cond = idx
        logits = model(idx_cond)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    return decode(idx[0].tolist())

if __name__ == "__main__":
    print(generate_text())
