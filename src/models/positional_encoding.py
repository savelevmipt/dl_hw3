import torch
import numpy as np


class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_size, maxlen):
        """
        emb_size - размер эмбеддингов
        maxlen - длина контекста
        """
        super(PositionalEncoding, self).__init__()

        zerospos = torch.zeros((maxlen, emb_size))
        position = torch.arange(0, maxlen).unsqueeze(1).float()
        dividing = torch.exp(-torch.arange(0, emb_size, 2).float() * np.log(10000.0) / emb_size)
        zerospos[:, 0::2] = torch.sin(position * dividing)
        zerospos[:, 1::2] = torch.cos(position * dividing)
        zerospos = zerospos.unsqueeze(0)
        self.register_buffer("pos_emb", zerospos)

    def forward(self, token_embedding):
        """
        token_embedding - тензор матрицы эмбеддингов
        """
        return token_embedding + self.pos_emb[:, : token_embedding.size(1)]