import torch
import torch.nn as nn

class EncoderOnlyTransformer(nn.Module):

  def __init__(self, vocab_size, embedding_dim, block_size, num_heads, num_blocks, dropout):
    super().__init__()

    self.block_size = block_size
    self.dropout = dropout

    self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
    self.position_embedding_table = nn.Embedding(block_size, embedding_dim)

    # self.sa_layers = nn.Sequential(*[Block(embedding_dim, num_heads, dropout, mode="encode") for _ in range(num_blocks)])
    self.sa_layers = nn.ModuleList([Block(embedding_dim, num_heads, dropout, mode="encode") for _ in range(num_blocks)])
    self.ln = nn.LayerNorm(embedding_dim)

  def forward(self, x, pad_id = None):
    '''
    x has shape (batch_size, sequence length <= block_size)
    '''
    token_embeddings = self.token_embedding_table(x)
    pos_embeddings = self.position_embedding_table(torch.arange(x.shape[1], device=x.device))

    out = token_embeddings + pos_embeddings

    if pad_id is None:
        x = None # forget x

    for layer in self.sa_layers:
      out = layer(out, x, pad_id) # (batch_size, block_size, embedding_dim)
    out = self.ln(out)

    return out
