import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from typing import Literal

def train():
  m = DecoderOnlyTransformer(vocab_size, embedding_dim, block_size, num_heads, num_blocks, dropout).to(device)
  optimizer = torch.optim.AdamW(m.parameters(), lr=lr)

  # Train loop
  for _ in tqdm(range(train_iters)):

    xb, yb = get_batch(train_data, batch_size, block_size)

    logits = m(xb.to(device))

    loss = F.cross_entropy(logits.reshape(batch_size * block_size, -1), yb.reshape(-1).to(device))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if _ % print_iters == 0:
      train_loss = loss.item()

      xb, yb = get_batch(val_data, batch_size, block_size)

      m.eval()

      with torch.no_grad():
        logits = m(xb.to(device))
        val_loss = F.cross_entropy(logits.reshape(batch_size * block_size, -1), yb.reshape(-1).to(device)).item()

      print(train_loss, val_loss)

      m.train()

if __name__ == "__main__":
  embedding_dim = 64*6
  num_heads = 6
  block_size = 256
  batch_size = 64
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  train_iters = 500
  print_iters = 500
  lr = 3e-4
  num_blocks = 6
  dropout = 0.2
  
  train()
