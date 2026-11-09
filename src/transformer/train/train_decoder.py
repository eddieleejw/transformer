import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

from typing import Literal

from transformer.models.decoder_only_transformer import DecoderOnlyTransformer
from transformer.data.data import get_batch

def main():

  with open('data/tinyshakespeare/input.txt', 'r') as f:
    text = f.read()

  chars = sorted(list(set(text)))
  stoi = {c:i for i,c in enumerate(chars)}
  itos = {i:c for i,c in enumerate(chars)}

  vocab_size = len(chars)
  encode = lambda s: [stoi[c] for c in s]
  decode = lambda l: ''.join([itos[i] for i in l])

  train_size = 0.9
  data = torch.tensor(encode(text), dtype=torch.long)
  n = int(train_size * len(data))
  train_data = data[:n]
  val_data = data[n:]

  embedding_dim = 64*6
  num_heads = 6
  block_size = 256
  batch_size = 64
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  train_iters = 5000
  print_iters = 500
  lr = 3e-4
  num_blocks = 6
  dropout = 0.2
  artifact_dir = Path("artifacts")

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

  torch.save(
    {
      "model": m.state_dict(),
      "optim": optimizer.state_dict(),
      "epoch": train_iters,
      "loss": loss.item()
    },
    artifact_dir / "decoder_checkpoint.pth"
  )


if __name__ == "__main__":
  main()