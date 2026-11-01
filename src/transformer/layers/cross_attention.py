class MultiHeadCrossAttention(nn.Module):

  def __init__(
    self,
    x_embedding_dim,
    y_embedding_dim,
    num_heads,
    head_dim,
    value_dim,
    dropout,
  ):
    super().__init__()

    self.x_embedding_dim = x_embedding_dim
    self.y_embedding_dim = y_embedding_dim
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.value_dim = value_dim

    self.query_matrix = nn.Linear(x_embedding_dim, num_heads * head_dim)
    self.key_matrix = nn.Linear(y_embedding_dim, num_heads * head_dim)
    self.value_matrix = nn.Linear(y_embedding_dim, num_heads * value_dim)

    self.proj = nn.Linear(num_heads * value_dim, x_embedding_dim)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

  def forward(self, x, y):
    '''
    x has shape (batch_size, x_block_size, x_embedding_dim)
    y has shape (batch_size, y_block_size, y_embedding_dim)
    '''
    x_batch_size, x_block_size, x_embedding_dim = x.shape
    y_batch_size, y_block_size, y_embedding_dim = y.shape

    assert x_embedding_dim == self.x_embedding_dim, f"Dimension of x ({x_embedding_dim}) doesn't match x embedding dimension of multihead attention block ({self.x_embedding_dim})"
    assert y_embedding_dim == self.y_embedding_dim, f"Dimension of y ({y_embedding_dim}) doesn't match y embedding dimension of multihead attention block ({self.y_embedding_dim})"
    assert x_batch_size == y_batch_size, f"Batch size of x ({x_batch_size}) does not match y ({y_batch_size})"

    batch_size = x_batch_size

    q_vec = self.query_matrix(x) # (batch_size, x_block_size, num_heads * head_dim)
    k_vec = self.key_matrix(y) # (batch_size, y_block_size, num_heads * head_dim)
    v_vec = self.value_matrix(y) # (batch_size, y_block_size, num_heads * value_dim)

    q_vec = q_vec.reshape(batch_size, x_block_size, self.num_heads, self.head_dim) # (batch_size, x_block_size, num_heads, head_dim)
    q_vec = q_vec.transpose(1, 2) # (batch_size, num_heads, x_block_size, head_dim)

    k_vec = k_vec.reshape(batch_size, y_block_size, self.num_heads, self.head_dim) # (batch_size, y_block_size, num_heads, head_dim)
    k_vec = k_vec.transpose(1, 2) # (batch_size, num_heads, y_block_size, head_dim)

    v_vec = v_vec.reshape(batch_size, y_block_size, self.num_heads, self.value_dim) # (batch_size, y_block_size, num_heads, value_dim)
    v_vec = v_vec.transpose(1, 2) # (batch_size, num_heads, y_block_size, value_dim)

    weights = (q_vec @ k_vec.transpose(-2, -1))  * (self.head_dim ** -0.5) # (batch_size, num_heads, x_block_size, y_block_size)

    weights = torch.softmax(weights, dim = -1)
    weights = self.dropout1(weights)

    out = weights @ v_vec # (batch_size, num_heads, x_block_size, value_dim)
    out = out.transpose(1, 2) # (batch_size, x_block_size, num_heads, value_dim)
    out = out.reshape(batch_size, x_block_size, -1) # (batch_size, x_block_size, num_heads * value_dim)

    out = self.proj(out) # (batch_size, x_block_size, x_embedding_dim)

    out = self.dropout2(out)

    return out
