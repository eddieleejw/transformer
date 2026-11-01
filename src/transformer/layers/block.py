class Block(nn.Module):

  def __init__(self, embedding_dim, num_heads, dropout, mode):
    super().__init__()
    self.sa = MultiHeadAttention(embedding_dim, num_heads, dropout, mode = mode)
    self.ffwd = FeedForward(embedding_dim, dropout)
    self.ln1 = nn.LayerNorm(embedding_dim)
    self.ln2 = nn.LayerNorm(embedding_dim)

  def forward(self, x):
    '''
    x has shape (batch_size, block_size, embedding_dim)

    output has shape (batch_size, block_size, embedding_dim)
    '''
    x = x + self.sa(self.ln1(x)) # skip connection
    x = x + self.ffwd(self.ln2(x))

    return x
