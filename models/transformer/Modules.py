import torch
import torch.nn as nn
import numpy as np
from heatmap import heatmap
__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        # print('attn', attn.shape)
        attn = attn / self.temperature
        # print('attn', attn.shape)
        # print('k',k.shape)
        # print('q',q.shape)
        # heatmap(attn, k)



        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        # heatmap(attn, attn)
        attn = self.softmax(attn) #[80,7,200] [80,7,7]
        # heatmap(attn, attn)

        attn = self.dropout(attn) #[80,7,200] [80,7,7]

        output = torch.bmm(attn, v)
        # heatmap(attn, attn)

        return output, attn
