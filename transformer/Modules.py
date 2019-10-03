import torch
import torch.nn as nn
import numpy as np

__author__ = "Yu-Hsiang Huang"


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=3)

    def forward(self, q, k, v, mask=None):

        bq, nq, lq, iq = q.size()
        bk, nk, lk, ik = k.size()
        bv, nv, lv, iv = v.size()

        attn = torch.bmm(q.view(-1, lq, iq), k.transpose(2, 3).view(-1, ik, lk)).view(bq, nq, lq, lk)
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)

        if mask is not None:
            attn = attn.masked_fill(mask, 0.)

        attn = self.dropout(attn)
        output = torch.bmm(attn.view(-1, lq, lk), v.view(-1, lv, iv)).view(bq, nq, lq, iv)

        return output, attn
