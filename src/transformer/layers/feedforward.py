class FeedForward(nn.Module):

  def __init__(self, embedding_dim, dropout):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(embedding_dim, 4 * embedding_dim),
        nn.ReLU(),
        nn.Linear(embedding_dim * 4, embedding_dim),
        nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.layers(x)
