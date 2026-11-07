import torch

def get_batch(data: torch.tensor, batch_size: int, block_size: int) -> tuple[torch.tensor, torch.tensor]:
  '''
  data has shape (n), and each value is an encoded token (i.e. an integer in [0, vocab_size))
  '''
  idx = torch.randint(len(data)-block_size, (batch_size,))
  x = torch.stack([data[id:id+block_size] for id in idx])
  y = torch.stack([data[id+1:id+block_size+1] for id in idx])

  return x,y
