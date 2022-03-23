# large parts of this code are taken from @yashbonde/rasp
# which is in torch modified version of @karpathy/minGPT

from math import ceil
from functools import partial
from random import randrange
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat

# functions


# main classes
class XcitAttention(nn.Module):
    # from:
    # https://github.com/facebookresearch/xcit/blob/3bd7bf7f483aea18c63fbc242e4f27d3da486c17/xcit.py#L221
    def __init__(self, dim, n_head, seq_len, drop=0.0, qkv_bias=False, temp=0.8):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.split_size = dim
        self.temperature = temp
        self.n_head = n_head

    def forward(self, x, attn_mask):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_head, C // self.n_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        if attn_mask is not None:
            attn.masked_fill_(~attn_mask, -1e6)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, dim, n_head, seq_len, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.split_size = dim

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", ~torch.ones(dim // n_head, dim // n_head).triu_(1).bool())

        self.attn = XcitAttention(dim, n_head, seq_len)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.mask)
        x = x + self.mlp(self.ln2(x))
        return x


class CrossCovarianceTestTransformer(nn.Module):
    def __init__(self, dim=16, depth=3, n_head=4, seq_len=128, vocab_size=39, dropout=0.0):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, dim))
        self.drop = nn.Dropout(dropout)

        # transformer
        self.blocks = nn.ModuleList([Block(dim, n_head, seq_len, dropout) for _ in range(depth)])

        # decoder head
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        self.seq_len = seq_len  # as required by AutoregressiveWrapper

    def forward(self, x):
        # forward the GPT model
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        b, t = x.size()
        token_embeddings = self.tok_emb(x)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.head(x)

        return logits
