import torch
import torch.nn as nn

from transformer.layers.self_attention import MultiHeadAttention
from transformer.layers.feedforward import FeedForward

class Block(nn.Module):

  def __init__(self, embedding_dim, num_heads, dropout, mode):
    super().__init__()
    self.sa = MultiHeadAttention(embedding_dim, num_heads, dropout, mode = mode)
    self.ffwd = FeedForward(embedding_dim, dropout)
    self.ln1 = nn.LayerNorm(embedding_dim)
    self.ln2 = nn.LayerNorm(embedding_dim)

  def forward(self, x, key_tokens = None, pad_id = None):
    '''
    x has shape (batch_size, block_size, embedding_dim)

    output has shape (batch_size, block_size, embedding_dim)
    '''
    x = x + self.sa(self.ln1(x), key_tokens, pad_id) # skip connection
    x = x + self.ffwd(self.ln2(x))

    return x