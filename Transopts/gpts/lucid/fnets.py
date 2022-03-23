# large parts of this code are taken from @yashbonde/rasp
# which is in torch modified version of @karpathy/minGPT

from math import ceil
from functools import partial
from random import randrange
import torch
import torch.nn.functional as F
from torch import nn, einsum

import einops as ein

# functions


# main classes
class Fourier(nn.Module):
    # https://arxiv.org/pdf/2105.03824.pdf
    # These are the differences that we have compared to the paper implementation
    # Since in the paper they have encoder only architecture, this creates a problem
    # where it is hard to implement casual attention
    # there is one bypass as follows:
    # ```
    # embd = torch.randn(1, 16, 18) # emebdding with shape [b,n,e]
    # e = ein.repeat(embd, "b n e -> b n h e", h = embd.shape[1]) # tile the values for casual
    # e.masked_fill_(torch.ones(1, 16, 16, 18).triu_(1).bool(), 0,) # casual attn
    # out = torch.fft.fft(e).real # calculate the FFT
    # out = out[:, torch.arange(16), torch.arange(16), :] # gather the diagonal elements
    # ```
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        device = x.device
        B, N, C = x.shape
        e = ein.repeat(x, "b n c -> b n h c", h=N)
        e.masked_fill_(
            torch.ones(1, N, N, 1).triu_(1).bool().to(device),
            0,
        )  # casual attn
        out = torch.fft.fft(e).real
        x = out[:, torch.arange(N), torch.arange(N), :]  # gather the diagonal elements
        x = self.ln(x)
        return x


class Block(nn.Module):
    # now in the paper they used an encoder only architecture
    # while we are trying to morph this thing for decoder only
    # it requires the following changes (from paper persp.):
    # - pre-norm instead of post norm

    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.split_size = dim
        self.attn = Fourier(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class FourierGPT(nn.Module):
    def __init__(self, dim=16, depth=3, seq_len=128, vocab_size=39, dropout=0.0):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, dim))
        self.drop = nn.Dropout(dropout)

        # transformer
        self.blocks = nn.ModuleList([Block(dim, dropout) for _ in range(depth)])

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
