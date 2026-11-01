class EncoderOnlyTransformer(nn.Module):

  def __init__(self, vocab_size, embedding_dim, block_size, num_heads, num_blocks, dropout):
    super().__init__()

    self.block_size = block_size
    self.dropout = dropout

    self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
    self.position_embedding_table = nn.Embedding(block_size, embedding_dim)

    self.sa_layers = nn.Sequential(*[Block(embedding_dim, num_heads, dropout, mode="encode") for _ in range(num_blocks)])
    self.ln = nn.LayerNorm(embedding_dim)

  def forward(self, x):
    '''
    x has shape (batch_size, sequence length <= block_size)
    '''
    token_embeddings = self.token_embedding_table(x)
    pos_embeddings = self.position_embedding_table(torch.arange(x.shape[1], device=x.device))

    out = token_embeddings + pos_embeddings
    out = self.sa_layers(out) # (batch_size, block_size, embedding_dim)
    out = self.ln(out)

    return out
