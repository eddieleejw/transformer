import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

def get_batches(batch_size, sources, translations, de_stoi, en_stoi):
  indices = torch.randint(len(sources), (batch_size,)).tolist()

  y = pad_sequence(
      [sources[i] for i in indices],
      batch_first = True,
      padding_value = de_stoi["<PAD>"]
      )

  x_pre = pad_sequence(
      [translations[i] for i in indices],
      batch_first = True,
      padding_value = en_stoi["<PAD>"]
  )

  x = x_pre[:,:-1]

  target = x_pre[:,1:]

  return x, y, target