import torch
import torch.nn as nn


class Word2vecCBOW(nn.Module):
    """Word2vec continuous bag of words (CBOW) model."""

    def __init__(self, num_embeddings: int, embedding_dim: int=300):
        # Initialise super
        super().__init__()
        # Set layers
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.linear = nn.Linear(embedding_dim, num_embeddings)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Retrieve embedding
        embedding = self.embedding(X)
        # Sum values
        sum = embedding.sum(dim=1)
        # Predict value
        return self.linear(sum)


class Word2vecSkipgram(nn.Module):
    """Word2vec Skipgram model."""

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int = 300,
            predictions: int = 10,
        ):
        # Initialise super
        super().__init__()
        # Set predictions
        self.predictions = predictions
        # Set layers
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.linear = nn.Linear(embedding_dim, num_embeddings*predictions)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Retrieve embedding
        embedding = self.embedding(X)
        # Compute output layer
        out = self.linear(embedding).reshape(X.shape[0], self.predictions, -1)
        # Compute softmax and return
        return out.softmax(-1)
