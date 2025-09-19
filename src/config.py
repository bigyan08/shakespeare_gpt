import torch


#dataset
DATA_PATH='data/input.txt'


#model
VOCAB_SIZE = None   #set after loading dataset
D_MODEL=512
NUM_HEADS=4
HIDDEN_LAYER=512
NUM_DEC=4
BLOCK_SIZE=128
DROPOUT=0.1

#training
BATCH_SIZE=64
EPOCHS=1
LEARNING_RATE=3e-4

#device setup
# DEVICE='cuda' if torch.cuda.is_available else 'cpu'
DEVICE = 'cpu'