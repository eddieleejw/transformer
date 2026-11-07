class DecoderOnlyTransformer(nn.Module):

  def __init__(self, vocab_size, embedding_dim, block_size, num_heads, num_blocks, dropout):
    super().__init__()

    self.block_size = block_size
    self.dropout = dropout

    self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
    self.position_embedding_table = nn.Embedding(block_size, embedding_dim)

    # self.sa_layers = nn.Sequential(*[Block(embedding_dim, num_heads, dropout, mode="decode") for _ in range(num_blocks)])
    self.sa_layers = nn.ModuleList([Block(embedding_dim, num_heads, dropout, mode="decode") for _ in range(num_blocks)])
    self.ln = nn.LayerNorm(embedding_dim)
    self.ll = nn.Linear(embedding_dim, vocab_size)

  def forward(self, x, pad_id = None):
    '''
    x has shape (batch_size, sequence length <= block_size)
    '''
    token_embeddings = self.token_embedding_table(x)
    pos_embeddings = self.position_embedding_table(torch.arange(x.shape[1], device=x.device))

    out = token_embeddings + pos_embeddings

    if pad_id is None:
      x = None # forget x

    # out = self.sa_layers(out) # (batch_size, block_size, embedding_dim)
    for layer in self.sa_layers:
      out = layer(out, x, pad_id)
    out = self.ln(out)
    out = self.ll(out) # (batch_size, block_size, vocab_size)

    return out

  @torch.no_grad()
  def generate(self, x, generation_length: int):
    '''
    x has shape (batch_size, sequence length)
    '''
    for _ in range(generation_length):
      idx = x[:, -self.block_size:]
      logits = self.forward(idx) # (batch_size, block_size, vocab_size)
      last_logits = logits[:, -1, :] # (batch_size, vocab_size)
      weights = torch.softmax(last_logits, dim = -1)
      next_idx = torch.multinomial(weights, 1)
      x = torch.cat([x, next_idx], dim = 1)

    return x
