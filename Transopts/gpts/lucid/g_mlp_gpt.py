from math import ceil
from functools import partial
from random import randrange
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat

import torch
import torch.nn as nn
from operator import itemgetter
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states


# for routing arguments into the functions of the reversible layer
def route_args(router, args, depth):
    routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router]

    for key in matched_keys:
        val = args[key]
        for depth, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
            new_f_args, new_g_args = map(lambda route: ({key: val} if route else {}), routes)
            routed_args[depth] = ({**f_args, **new_f_args}, {**g_args, **new_g_args})
    return routed_args


def layer_drop(layers, prob):
    to_drop = torch.empty(len(layers)).uniform_(0, 1) < prob
    blocks = [block for block, drop in zip(layers, to_drop) if not drop]
    blocks = layers[:1] if len(blocks) == 0 else blocks
    return blocks


# following example for saving and setting rng here https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html
class Deterministic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng=False, set_rng=False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)


# heavily inspired by https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
# once multi-GPU is confirmed working, refactor and send PR back to source
class ReversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(self, x, f_args={}, g_args={}):
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1, y2 = None, None

        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)

        return torch.cat([y1, y2], dim=2)

    def backward_pass(self, y, dy, f_args={}, g_args={}):
        y1, y2 = torch.chunk(y, 2, dim=2)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim=2)
        del dy

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim=2)
            dx = torch.cat([dx1, dx2], dim=2)

        return x, dx


class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, args):
        ctx.args = args
        for block, kwarg in zip(blocks, args):
            x = block(x, **kwarg)
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        args = ctx.args
        for block, kwargs in zip(ctx.blocks[::-1], args[::-1]):
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None


class SequentialSequence(nn.Module):
    def __init__(self, layers, args_route={}, layer_dropout=0.0):
        super().__init__()
        assert all(
            len(route) == len(layers) for route in args_route.values()
        ), "each argument route map must have the same depth as the number of sequential layers"
        self.layers = layers
        self.args_route = args_route
        self.layer_dropout = layer_dropout

    def forward(self, x, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        if self.training and self.layer_dropout > 0:
            layers_and_args = layer_drop(layers_and_args, self.layer_dropout)

        for (f,), (f_args, _) in layers_and_args:
            x = x + f(x, **f_args)
        return x


class ReversibleSequence(nn.Module):
    def __init__(self, blocks, args_route={}, layer_dropout=0.0):
        super().__init__()
        self.args_route = args_route
        self.layer_dropout = layer_dropout
        self.blocks = nn.ModuleList([ReversibleBlock(f=f, g=g) for f, g in blocks])

    def forward(self, x, layer_dropout=0.0, **kwargs):
        x = torch.cat([x, x], dim=-1)

        blocks = self.blocks
        args = route_args(self.args_route, kwargs, len(blocks))
        args = list(map(lambda x: {"f_args": x[0], "g_args": x[1]}, args))

        layers_and_args = list(zip(blocks, args))

        if self.training and layer_dropout > 0:
            layers_and_args = layer_drop(layers_and_args, layer_dropout)
            blocks, args = map(lambda ind: list(map(itemgetter(ind), layers_and_args)), (0, 1))

        out = _ReversibleFunction.apply(x, blocks, args)
        return torch.stack(out.chunk(2, dim=-1)).sum(dim=0)


# functions


def exists(val):
    return val is not None


def cast_tuple(val, num):
    return ((val,) * num) if not isinstance(val, tuple) else val


def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return tensor
    remainder = ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value=value)


def dropout_layers(layers, prob_survival):
    if prob_survival == 1:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0.0, 1.0) > prob_survival

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers


# helper classes


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner_dim = int(dim * mult * 2 / 3)

        self.net = nn.Sequential(nn.Linear(dim, inner_dim * 2), GEGLU(), nn.Linear(inner_dim, dim))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias=False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        mask = torch.ones(sim.shape[-2:], device=device).triu(1).bool()
        sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)

        attn = sim.softmax(dim=-1)
        out = einsum("b i j, b j d -> b i d", attn, v)
        return self.to_out(out)


class LocalAttention(nn.Module):
    def __init__(self, dim_in, dim_inner, dim_out, window=128):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.window = window

        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias=False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        b, n, *_, device, w = *x.shape, x.device, self.window

        x = pad_to_multiple(x, w, dim=-2, value=0.0)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        window_fn = lambda t: rearrange(t, "b (w n) d -> b w n d", n=w)
        q, k, v = map(window_fn, (q, k, v))

        k, v = map(lambda t: F.pad(t, (0, 0, 0, 0, 1, 0)), (k, v))
        k, v = map(lambda t: torch.cat((k[:, :-1], k[:, 1:]), dim=2), (k, v))

        sim = einsum("b w i d, b w j d -> b w i j", q, k) * self.scale
        buckets, i, j = sim.shape[-3:]

        mask_value = -torch.finfo(sim.dtype).max
        mask = torch.ones(i, j, device=device).triu_(j - i + 1).bool()
        mask = repeat(mask, "i j -> () u i j", u=buckets)

        sim.masked_fill_(mask, mask_value)

        attn = sim.softmax(dim=-1)

        out = einsum("b w i j, b w j d -> b w i d", attn, v)
        out = rearrange(out, "b w n d -> b (w n) d")
        out = self.to_out(out[:, :n])
        return out


class CausalSGU(nn.Module):
    def __init__(self, dim, dim_seq, init_eps=1e-3, heads=4, act=nn.Identity()):
        super().__init__()
        dim_out = dim // 2

        self.norm = nn.LayerNorm(dim_out)

        self.heads = heads
        self.weight = nn.Parameter(torch.zeros(heads, dim_seq, dim_seq))
        self.bias = nn.Parameter(torch.zeros(heads, dim_seq))

        init_eps /= dim_seq
        nn.init.uniform_(self.weight, -init_eps, init_eps)
        nn.init.constant_(self.bias, 1.0)

        self.act = act
        self.register_buffer("mask", ~torch.ones(dim_seq, dim_seq).triu_(1).bool())

    def forward(self, x, gate_res=None):
        device, n, h = x.device, x.shape[1], self.heads

        res, gate = x.chunk(2, dim=-1)
        gate = self.norm(gate)

        weight, bias = self.weight, self.bias
        weight, bias = weight[:, :n, :n], bias[:, :n]

        weight = weight * self.mask[None, :n, :n].int().float()

        gate = rearrange(gate, "b n (h d) -> b h n d", h=h)
        gate = einsum("b h n d, h m n -> b h m d", gate, weight)
        gate = gate + rearrange(bias, "h n -> () h n ()")
        gate = rearrange(gate, "b h n d -> b n (h d)")

        if exists(gate_res):
            gate = gate + gate_res

        return self.act(gate) * res


class CausalLocalSGU(nn.Module):
    def __init__(self, dim, dim_seq, init_eps=1e-3, heads=4, window=128, act=nn.Identity()):
        super().__init__()
        dim_out = dim // 2

        self.norm = nn.LayerNorm(dim_out)

        self.heads = heads
        self.window = window
        self.weight = nn.Parameter(torch.zeros(heads, window, window * 2))
        self.bias = nn.Parameter(torch.zeros(heads, window))

        init_eps /= window
        nn.init.uniform_(self.weight, -init_eps, init_eps)
        nn.init.constant_(self.bias, 1.0)

        self.act = act
        self.register_buffer("mask", ~torch.ones(window, window * 2).triu_(window + 1).bool())

    def forward(self, x, gate_res=None):
        device, n, h, w = x.device, x.shape[1], self.heads, self.window

        res, gate = x.chunk(2, dim=-1)

        gate = pad_to_multiple(gate, w, dim=-2)
        gate = rearrange(gate, "b (w n) d -> b w n d", n=w)

        gate = self.norm(gate)

        gate = F.pad(gate, (0, 0, 0, 0, 1, 0), value=0.0)
        gate = torch.cat((gate[:, :-1], gate[:, 1:]), dim=2)

        weight, bias = self.weight, self.bias

        weight = weight * self.mask[None, ...].int().float()

        gate = rearrange(gate, "b w n (h d) -> b w h n d", h=h)
        gate = einsum("b w h n d, h m n -> b w h m d", gate, weight)
        gate = gate + rearrange(bias, "h n -> () () h n ()")

        gate = rearrange(gate, "b w h n d -> b w n (h d)")

        gate = rearrange(gate, "b w n d -> b (w n) d")
        gate = gate[:, :n]

        if exists(gate_res):
            gate = gate + gate_res

        return self.act(gate) * res


class AxiallyFold(nn.Module):
    def __init__(self, dim, every, fn):
        super().__init__()
        self.fn = fn
        self.every = every
        self.conv = nn.Conv1d(dim, dim, kernel_size=every, groups=dim) if every > 1 else None

    def forward(self, x):
        every = self.every
        if every <= 1:
            return self.fn(x)

        n = x.shape[1]
        x = pad_to_multiple(x, self.every, dim=-2)
        x = rearrange(x, "b (n e) d -> (b e) n d", e=every)
        x = self.fn(x)

        x = rearrange(x, "(b e) n d -> b d (n e)", e=every)
        x = F.pad(x, (every - 1, 0), value=0)
        out = self.conv(x)
        out = rearrange(out, "b d n -> b n d")
        return out[:, :n]


class gMLPBlock(nn.Module):
    def __init__(self, *, dim, seq_len, dim_ff, heads=4, causal=False, window=None, attn_dim=None, act=nn.Identity()):
        super().__init__()
        is_windowed = exists(window) and window < seq_len

        SGU_klass = partial(CausalLocalSGU, window=window) if is_windowed else CausalSGU
        Attention_klass = partial(LocalAttention, window=window) if is_windowed else Attention

        self.attn = Attention_klass(dim_in=dim, dim_inner=attn_dim, dim_out=dim_ff // 2) if exists(attn_dim) else None

        self.proj_in = nn.Sequential(nn.Linear(dim, dim_ff), nn.GELU())
        self.sgu = SGU_klass(dim_ff, seq_len, causal, heads=heads, act=act)
        self.proj_out = nn.Linear(dim_ff // 2, dim)

    def forward(self, x):
        gate_res = self.attn(x) if exists(self.attn) else None
        x = self.proj_in(x)
        x = self.sgu(x, gate_res=gate_res)
        x = self.proj_out(x)
        return x


# main classes


class gMLPGPT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        vocab_size=39,
        seq_len=128,
        n_head=1,
        ff_mult=4,
        prob_survival=1.0,
        reversible=False,
        window=None,
        attn_dim=None,
        act=nn.Identity(),
    ):
        super().__init__()
        dim_ff = dim * ff_mult
        self.seq_len = seq_len
        self.prob_survival = prob_survival

        self.to_embed = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.randn(seq_len, dim), requires_grad=True)

        window = cast_tuple(window, depth)
        window = tuple(map(lambda t: t if isinstance(t, tuple) else (t, 1), window))

        attn_dims = cast_tuple(attn_dim, depth)

        assert len(window) == depth, f"num window sizes {len(window)} must be equal to depth {depth}"

        layers = nn.ModuleList([])

        for ind, (w, ax), attn_dim in zip(range(depth), window, attn_dims):
            attn_dim = attn_dim if exists(window) else None
            get_gmlp = lambda: PreNorm(
                dim,
                AxiallyFold(
                    dim, ax, gMLPBlock(dim=dim, dim_ff=dim_ff, seq_len=seq_len, heads=n_head, window=w, act=act, attn_dim=attn_dim)
                ),
            )

            layer_blocks = nn.ModuleList([get_gmlp()])

            if reversible:
                layer_blocks.append(FeedForward(dim, mult=ff_mult))

            layers.append(layer_blocks)

        execute_klass = SequentialSequence if not reversible else ReversibleSequence
        self.net = execute_klass(layers)

        self.to_logits = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, vocab_size))

    def forward(self, x):
        layer_dropout = 1.0 - self.prob_survival

        x = self.to_embed(x) + self.pos_emb[: x.size(1)]
        out = self.net(x, layer_dropout=layer_dropout)
        return self.to_logits(out)
