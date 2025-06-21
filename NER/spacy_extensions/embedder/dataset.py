from torch.utils.data import Dataset
from torch.nn.functional import pad
from typing import List, Tuple
import numpy as np
import torch


class ContextDataset(Dataset):

    def __init__(
            self,
            data: List[List[int]],
            context: Tuple[int, int],
            pad : int,
        ):
        """Create context dataset"""
        self.data = [torch.tensor(sent, dtype=int) for sent in data]
        self.context = context
        self._lengths = np.asarray([x.shape[0] for x in self.data], dtype=int)
        self._lengths = np.cumsum(self._lengths)

        # Prepare output
        self.output = torch.full((sum(self.context),), pad, dtype=int)

    def __len__(self) -> int:
        """Length of ContextDataset."""
        return self._lengths[-1]

    def __getitem__(self, index) -> torch.Tensor:
        """Retrieve item from dataset."""
        # Raise index error in case index is out of bounds
        if not 0 <= index < len(self):
            raise IndexError(index)
            
        # Compute sent index item
        sent_index = np.nonzero(self._lengths <= index)[0]
        if sent_index.shape[0] == 0:
            sent_index = 0
        else:
            sent_index = sent_index[-1]+1
            index -= self._lengths[sent_index-1]

        # Retrieve sentence
        sent = self.data[sent_index]
        # Retrieve sentence boundaries
        left = max(0, index - self.context[0])
        right = min(sent.shape[0], index + self.context[1])+1
        # Get left and right context
        left = sent[left : index]
        right = sent[index+1 : right]

        # Initialise output
        output = self.output.detach().clone()
        output[self.context[0] - left .shape[0] : self.context[0]] = left
        output[self.context[0] : self.context[0] + right.shape[0]] = right

        # Return result
        return output, sent[index]