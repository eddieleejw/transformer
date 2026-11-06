class EncoderDecoderBlock(nn.Module):

  def __init__(self, x_embedding_dim, num_sa_heads, dropout, ca_args: list):
    super().__init__()
    self.sa = MultiHeadAttention(x_embedding_dim, num_sa_heads, dropout, mode = "decode")
    self.ca = MultiHeadCrossAttention(*ca_args)
    self.ffwd = FeedForward(x_embedding_dim, dropout)
    self.ln1 = nn.LayerNorm(x_embedding_dim)
    self.ln2 = nn.LayerNorm(x_embedding_dim)
    self.ln3 = nn.LayerNorm(x_embedding_dim)

  def forward(self, x, y, x_key_tokens = None, x_pad_id = None, y_key_tokens = None, y_pad_id = None):
    '''
    x has shape (batch_size, x_block_size, x_embedding_dim)

    y is the output of the encoder section of the transformer and has shape (batch_size, y_block_size, y_embedding_dim)

    output has shape (batch_size, x_block_size, x_embedding_dim)
    '''
    x = x + self.sa(self.ln1(x), x_key_tokens, x_pad_id) # skip connection
    x = x + self.ca(self.ln2(x), y, y_key_tokens, y_pad_id)
    x = x + self.ffwd(self.ln3(x))

    return x