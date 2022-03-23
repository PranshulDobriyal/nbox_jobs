#!/usr/bin/env python3

# large parts of this code are taken from @yashbonde/rasp
# which is in turn modified version of @karpathy/minGPT
# and from @lucidrains/learning_to_expire
#
# definations of einops variables
# b: batch_size
# n: number of tokens in sequence
# i: source length
# j: target length
# h: number of heads
#

import torch
import torch.nn.functional as F
from torch import nn, einsum

from collections import namedtuple
import einops as ein

Memory = namedtuple("Memory", ["mems", "times"])


def exists(val):
    return val is not None


def safe_cat(tensors, dim=-1):
    tensors = list(filter(exists, tensors))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim=dim)


def safe_add(tensor, n):
    if not exists(tensor):
        return None
    return tensor + n


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        n, device = x.shape[1], x.device
        t = torch.arange(n - 1, -1, -1, device=device).type_as(self.inv_freq)
        sinusoid_inp = einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb


class ExpireSpan(nn.Module):
    def __init__(self, dim, max_mem_len, ramp_len):
        super(ExpireSpan, self).__init__()
        self.max_mem_len = max_mem_len
        self.ramp_len = ramp_len
        self.to_expiration = nn.Linear(dim, 1)

        # as done in FB's code
        self.to_expiration.weight.data.fill_(0)
        nn.init.constant_(self.to_expiration.bias.data, val=-self.max_mem_len)

    def forward(self, mem, time, seq_len):
        exps = self.to_expiration(mem).squeeze(-1).sigmoid() * self.max_mem_len
        exps = ein.rearrange(exps, "b j -> b () () j")
        t = ein.rearrange(time, "b j -> b () () j")
        r = F.pad(exps - t, (0, seq_len), value=1.0)
        mask = torch.clamp((r / self.ramp_len) + 1, min=0.0, max=1.0)
        return exps, mask


class Attention(nn.Module):
    def __init__(self, dim, heads, drop=0):
        super(Attention, self).__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        dim_head = dim // heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.pos_proj = nn.Linear(dim, dim_head)

    def forward(self, x, pos_emb, mem=None, expire_mask=None):
        q = self.q_proj(x)

        if mem is not None:
            mem_len = mem.shape[1]
            context = torch.cat([mem, x], dim=1)
        else:
            mem_len = 0
            context = x

        kv = self.kv_proj(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: ein.rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, *kv))
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # calculate relative positional embeddings
        pos = self.pos_proj(pos_emb)
        pos_dots = einsum("b h i d, j d -> b h i j", q, pos) * self.scale

        # this part is from the function `rel_shift()`
        b, h, i, j = pos_dots.shape
        zero_pad = torch.zeros((b, h, i, 1), device=pos_dots.device).to(pos_dots.dtype)
        concatted = torch.cat([zero_pad, pos_dots], dim=-1)
        shifted = concatted.view(b, h, j + 1, i)[:, :, 1:]
        pos_dots = shifted.view_as(pos_dots)

        pos_dots = F.pad(pos_dots, (mem_len, 0), value=0)

        # add the to dot products and apply casual masking
        attn = dots + pos_dots
        mask = torch.ones(dots.shape[-2:], device=x.device).triu_(mem_len + 1).bool()
        mask = ein.rearrange(mask, "i j -> () () i j")
        attn.masked_fill_(mask, -1e10)
        del mask  # free memory

        # note that the attention matrix can go beyond the seq_len because we concatenate it
        # with memory.

        attn = attn.softmax(-1)  # [b h n e]
        if expire_mask != None:
            # pad expire_mask correctly
            expire_mask = F.pad(expire_mask, pad=(0, attn.shape[-1] - expire_mask.shape[-1]), value=1.0)
            attn = attn * expire_mask

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = ein.rearrange(out, " b h i d -> b i (h d)")
        out = self.proj_drop(self.out_proj(out))
        return out


class Block(nn.Module):
    def __init__(self, dim, max_mem_len, ramp_len, heads, seq_len, expire_loss_coef=1e-6, dropout=0.0):
        super().__init__()
        self.es = ExpireSpan(dim, max_mem_len, ramp_len)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

        self.seq_len = seq_len
        self.max_mem_len = max_mem_len
        self.ramp_len = ramp_len
        self.expire_loss_coef = expire_loss_coef

    def forward(self, x, mem, time, seq_len, pos_emb):
        b, n, e, dev = *x.shape, x.device
        exps, expire_mask = self.es(mem, time, seq_len) if mem is not None else (None, None)

        if self.training and time != None:
            forget_time_thres = torch.randint(0, self.max_mem_len, (b, 1), device=dev)
            forget_dropout_mask = (time < forget_time_thres).float()
            forget_dropout_mask = ein.rearrange(forget_dropout_mask, "b n -> b () () n")
            forget_dropout_mask = F.pad(forget_dropout_mask, (0, n), value=1.0)
            expire_mask *= forget_dropout_mask

        x = x + self.attn(x, pos_emb, mem, expire_mask)
        x = x + self.mlp(x)

        # calculate loss
        aux_loss = None
        if exps is not None:
            expiring_exps_mask = (expire_mask > 0) * (expire_mask < 1)
            span_loss = exps * expiring_exps_mask.float()[..., :-n]
            loss = span_loss.sum(-1).sum(-1).squeeze()
            aux_loss = loss / self.ramp_len / n

        return x, expire_mask, aux_loss


class ExpireSpanGPT(nn.Module):
    def __init__(
        self,
        dim=16,
        depth=3,
        heads=2,
        seq_len=128,
        max_mem_len=32,
        ramp_len=16,
        vocab_size=39,
        dropout=0.0,
        expire_loss_coef=1e-6,
    ):
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.max_mem_len = max_mem_len
        self.expire_loss_coef = expire_loss_coef
        self.seq_len = seq_len  # as required by AutoregressiveWrapper

        # input embedding stem
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = SinusoidalEmbedding(dim)
        self.drop = nn.Dropout(dropout)

        # transformer
        self.blocks = nn.ModuleList([Block(dim, max_mem_len, ramp_len, heads, seq_len, expire_loss_coef, dropout) for _ in range(depth)])

        # decoder head
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x, memory=None):
        # forward the GPT model
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        b, t = x.shape
        x = self.tok_emb(x)  # each index maps to a (learnable) vector
        pos_emb = self.pos_emb(x)

        hidden_states = []
        expire_mask_layers = []
        mems_layers = memory.mems if memory is not None else ((None,) * self.depth)
        times_layers = memory.times if memory is not None else ((None,) * self.depth)
        aux_loss = torch.tensor(0.0, requires_grad=True)

        for mem, time, blk in zip(mems_layers, times_layers, self.blocks):
            hidden_states.append(x)

            x, expire_mask, _aux_loss = blk(
                x=x,
                mem=mem,
                time=time,
                seq_len=t,
                pos_emb=pos_emb,
            )

            expire_mask_layers.append(expire_mask)  # add the new update masks
            if _aux_loss is not None:
                aux_loss = aux_loss + _aux_loss  # add the auxilary loss for this layer

        x = self.ln_f(x)
        logits = self.head(x)

        if self.seq_len == t:
            if expire_mask != None:
                mems_layers_new = []
                times_layers_new = []

                for mems, times, exp_mask in zip(mems_layers, times_layers, expire_mask_layers):
                    exp_mask = ein.rearrange(exp_mask, "b () () i -> b i")

                    # discard expired memories
                    expired_exps_mask = (exp_mask <= 0)[..., :-t]

                    # it is not possible to expire different amounts of memories across batches
                    # for now, will just expire the minimum of the expired memories across batches
                    num_to_expire = min(expired_exps_mask.sum(dim=-1))
                    _, indices = expired_exps_mask.float().topk(k=num_to_expire, dim=-1)
                    even_expired_exps_mask = torch.zeros_like(expired_exps_mask, device=x.device).scatter(-1, indices, 1.0).bool()

                    mems = mems.masked_select(~even_expired_exps_mask.unsqueeze(-1))
                    mems = mems.reshape(b, -1, self.dim)
                    mems_layers_new.append(mems)

                    times = times.masked_select(~even_expired_exps_mask)
                    times = times.reshape(b, -1)
                    times_layers_new.append(times)

                mems_layers = mems_layers_new
                times_layers = times_layers_new

            new_memories = map(lambda tensors: safe_cat(tensors, dim=1), list(zip(mems_layers, hidden_states)))
            new_memories = map(lambda tensor: tensor[:, -self.max_mem_len :].detach(), new_memories)

            new_times = torch.arange(t - 1, -1, -1, device=x.device)
            new_times = ein.repeat(new_times, "n -> b n", b=b)
            new_elapsed_times = map(lambda tensor: safe_cat((safe_add(tensor, t), new_times), dim=1), times_layers)
            new_elapsed_times = map(lambda tensor: tensor[-self.max_mem_len :], new_elapsed_times)

            memory = Memory(list(new_memories), list(new_elapsed_times))

        return logits, aux_loss, memory


if __name__ == "__main__":
    model = ExpireSpanGPT(seq_len=12, depth=1)

    print("-" * 20 + " Forward Pass <= seq_len " + "-" * 20)
    tokens = torch.randint(0, 39, (1, 8))
    logits, aux_loss, memory = model(tokens)
    print("logits.shape:", logits.shape)
    print("    aux_loss:", aux_loss)
    assert memory == None

    print("-" * 20 + " Forward Pass == seq_len w/o memory " + "-" * 20)
    tokens = torch.randint(0, 39, (1, 12))
    logits, aux_loss, memory = model(tokens)
    print("      logits.shape:", logits.shape)
    print("          aux_loss:", aux_loss)
    print(" memory.mems.shape:", [x.shape for x in memory.mems])
    print("memory.times.shape:", [x.shape for x in memory.times])

    print("-" * 20 + " Forward Pass == seq_len w/ memory " + "-" * 20)
    logits, aux_loss, memory = model(tokens, memory)
    print("      logits.shape:", logits.shape)
    print("          aux_loss:", aux_loss)
    print(" memory.mems.shape:", [x.shape for x in memory.mems])
    print("memory.times.shape:", [x.shape for x in memory.times])
