import torch
import torch.nn as nn


class ELMo(nn.Module):
    """Embedder from `Deep Contextualized Word Representations`
        DOI: https://aclanthology.org/N18-1202.
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
        self.bilstm1 = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = hidden_dim,
            batch_first = True,
            bidirectional = True,
        )
        self.bilstm2 = nn.LSTM(
            input_size = hidden_dim,
            hidden_size = hidden_dim,
            batch_first = True,
            bidirectional = True,
        )
        self.linear = nn.Linear(2 * hidden_dim, num_embeddings)
        self.softmax = nn.LogSoftmax(dim=-1)
    

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Apply embedding
        embedding = self.embedding(X)
        # Apply lstm layers
        bilstm1 = self.bilstm1(embedding)
        bilstm2 = self.bilstm2(bilstm1)
        # Embedding is concatenation
        print(bilstm1.shape)
        print(bilstm2.shape)