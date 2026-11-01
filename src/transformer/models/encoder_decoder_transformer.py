class EncoderDecoderTransformer(nn.Module):

  def __init__(self, x_embedding_dim, y_embedding_dim, x_block_size, y_block_size, x_vocab_size, y_vocab_size, num_enc_sa_heads, num_enc_blocks, num_dec_blocks, num_dec_sa_heads, num_dec_ca_heads, dec_ca_head_dim, dec_ca_value_dim, dropout):
    super().__init__()

    self.x_block_size = x_block_size
    self.y_block_size = y_block_size

    self.x_embedding_dim = x_embedding_dim
    self.y_embedding_dim = y_embedding_dim

    self.encoder = EncoderOnlyTransformer(y_vocab_size, y_embedding_dim, y_block_size, num_enc_sa_heads, num_enc_blocks, dropout)

    ca_args = [x_embedding_dim, y_embedding_dim, num_dec_ca_heads, dec_ca_head_dim, dec_ca_value_dim, dropout]
    self.a_layers = nn.ModuleList(*[EncoderDecoderBlock(x_embedding_dim, num_dec_sa_heads, dropout, ca_args) for _ in range(num_dec_blocks)])

    self.token_embedding_table = nn.Embedding(x_vocab_size, x_embedding_dim)
    self.position_embedding_table = nn.Embedding(x_block_size, x_embedding_dim)

    self.ln = nn.LayerNorm(x_embedding_dim)
    self.proj = nn.Linear(x_embedding_dim, x_vocab_size)

  def forward(self, x, y):
    '''
    x has shape (batch_size, x_block_size) with values in [0, x_vocab_size)
    y has shape (batch_size, y_block_size) with values in [0, y_vocab_size)
    '''

    x_batch_size, x_block_size = x.shape
    y_batch_size, y_block_size = y.shape

    assert x_batch_size == y_batch_size, f"Batch size of x ({x_batch_size}) is not equal to y ({y_batch_size})"
    assert x_block_size <= self.x_block_size
    assert y_block_size <= self.y_block_size

    y = self.encoder(y) # (batch_size, y_block_size, y_embedding_dim)

    token_embeddings = self.token_embedding_table(x) # (batch_size, x_block_size, x_embedding_dim)
    position_embeddings = self.position_embedding_table(torch.arange(x_block_size, device=x.device)) # (batch_size, x_block_size, x_embedding_dim)

    out = token_embeddings + position_embeddings # (batch_size, x_block_size, x_embedding_dim)

    for layer in self.a_layers:
      out = layer(out, y) # (batch_size, x_block_size, x_embedding_dim)
    out = self.ln(out) # (batch_size, x_block_size, x_embedding_dim)
    out = self.proj(out) # (batch_size, x_block_size, x_vocab_size)

    return out   