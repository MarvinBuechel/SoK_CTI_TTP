import torch
import torch.nn as nn


class NLM(nn.Module):
    """Embedder from `A Neural Probabilistic Language Model`
        DOI: https://dl.acm.org/doi/10.5555/944919.944966.
        """

    def __init__(
            self,
            num_embeddings: int,
            context: int,
            embedding_dim: int = 300,
            hidden_dim: int = 256,
        ):
        # Initialise super
        super().__init__()
        # Set dimensions
        self.embedding_dim = embedding_dim
        self.context = context
        # Set layers
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.hidden1 = nn.Linear(embedding_dim * context, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, num_embeddings)
        self.hidden3 = nn.Linear(embedding_dim * context, num_embeddings)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Retrieve embedding
        embedding = self.embedding(X)
        # View as layer
        embedding = embedding.view(-1, self.embedding_dim * self.context)
        # Compute tanh output layer
        tanh = self.hidden1(embedding).tanh()
        # Compute output layer
        output = self.hidden2(tanh) + self.hidden3(embedding)
        # Compute softmax and return
        return self.softmax(output)
