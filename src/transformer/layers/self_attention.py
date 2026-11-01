import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class AttentionHead(nn.Module):

  def __init__(self, embedding_dim, head_dim):
    super().__init__()
    self.query_matrix = nn.Linear(embedding_dim, head_dim)
    self.key_matrix = nn.Linear(embedding_dim, head_dim)
    self.value_matrix = nn.Linear(embedding_dim, head_dim)

  def forward(self, x):
    '''
    x has dimension (batch_size, block_size, embedding_dim)
    '''
    batch_size, block_size, embedding_dim = x.shape

    query_vector = self.query_matrix(x) # (batch_size, block_size, head_dim)
    key_vector = self.key_matrix(x) # (batch_size, block_size, head_dim)
    value_vector = self.value_matrix(x) # (batch_size, block_size, head_dim)

    attention_pattern = query_vector @ key_vector.transpose(-1, -2) # (batch_size, block_size, block_size)

    attention_pattern = attention_pattern * (key_vector.shape[-1] ** -0.5)

    mask = torch.tril(torch.ones((block_size, block_size)))

    attention_pattern = attention_pattern.masked_fill(mask==0, -torch.inf) # (batch_size, block_size, block_size)

    attention_pattern = torch.softmax(attention_pattern, dim = -1)

    out = attention_pattern @ value_vector # (batch_size, block_size, head_dim)

    return out


class MultiHeadAttention(nn.Module):

  def __init__(self, embedding_dim, num_heads, dropout, mode: Literal["decode", "encode"]):
    super().__init__()

    assert embedding_dim % num_heads == 0, f"Embedding dimension ({embedding_dim}) must be divisible by number of heads ({num_heads})"
    assert mode in ["decode", "encode"], f"Mode ({mode}) must be in ['decode', 'encode']"

    self.mode = mode

    self.embedding_dim = embedding_dim
    self.num_heads = num_heads
    self.head_dim = embedding_dim // num_heads

    self.query_matrix = nn.Linear(embedding_dim, embedding_dim)
    self.key_matrix = nn.Linear(embedding_dim, embedding_dim)
    self.value_matrix = nn.Linear(embedding_dim, embedding_dim)

    self.proj = nn.Linear(embedding_dim, embedding_dim)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

  def forward(self, x):
    '''
    x has shape (batch_size, block_size, embedding_dim)
    '''
    batch_size, block_size, embedding_dim = x.shape

    assert embedding_dim == self.embedding_dim, f"Dimension of x ({embedding_dim}) doesn't match embedding dimension of multihead attention block ({self.embedding_dim})"

    q_vec = self.query_matrix(x) # (batch_size, block_size, embedding_dim)
    k_vec = self.key_matrix(x) # (batch_size, block_size, embedding_dim)
    v_vec = self.value_matrix(x) # (batch_size, block_size, embedding_dim)


    q_vec = q_vec.reshape(batch_size, block_size, self.num_heads, self.head_dim) # (batch_size, block_size, num_heads, head_dim) (split the array that looks like [head 1 embedding, head 2 embedding, ...] into [[head 1 embedding], [head 2 embedding], ...])
    k_vec = k_vec.reshape(batch_size, block_size, self.num_heads, self.head_dim) # (batch_size, block_size, num_heads, head_dim)
    v_vec = v_vec.reshape(batch_size, block_size, self.num_heads, self.head_dim) # (batch_size, block_size, num_heads, head_dim)

    q_vec = q_vec.transpose(1, 2) # (batch_size, num_heads, block_size, head_dim)
    k_vec = k_vec.transpose(1, 2) # (batch_size, num_heads, block_size, head_dim)
    v_vec = v_vec.transpose(1, 2) # (batch_size, num_heads, block_size, head_dim)

    weights = (q_vec @ k_vec.transpose(-2, -1))  * (self.head_dim ** -0.5) # (batch_size, num_heads, block_size, block_size)

    if self.mode == "decode":
      mask = torch.tril(torch.ones(block_size, block_size, device=x.device))
      weights = weights.masked_fill(mask==0, -torch.inf)

    weights = torch.softmax(weights, dim = -1) # (batch_size, num_heads, block_size, block_size)
    weights = self.dropout1(weights)

    out = weights @ v_vec # (batch_size, num_heads, block_size, head_dim)
    out = out.transpose(1, 2) # (batch_size, block_size, num_heads, head_dim)
    out = out.reshape(batch_size, block_size, -1) # (batch_size, block_size, embedding_dim)

    out = self.proj(out) # (batch_size, block_size, embedding_dim)

    out = self.dropout2(out)

    return out
