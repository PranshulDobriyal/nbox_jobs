# from lucidrains

import torch
from torch import nn
import torch.nn.functional as F

# helper function


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


# top k filtering


def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


class AutoregressiveWrapper(nn.Module):
    def __init__(self, net, ignore_index=-100, pad_value=0):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        if hasattr(net, "seq_len"):
            # this lis lucidrains models
            self.max_seq_len = net.seq_len
        else:
            if hasattr(net.config, "n_ctx"):
                self.max_seq_len = net.config.n_ctx
            elif hasattr(net.config, "max_position_embeddings"):
                self.max_seq_len = net.config.max_position_embeddings

    @torch.no_grad()
    @eval_decorator
    def generate(self, start_tokens, seq_len, eos_token=None, temperature=1.0, filter_thres=0.9, greedy=False, **kwargs):
        device = start_tokens.device
        memory = None  # memory is used with Expire Span Model
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        out = start_tokens

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len :]

            logits = self.net(x, **kwargs)

            try:
                logits = logits.logits
            except AttributeError:
                pass

            if isinstance(logits, tuple):
                # Check if we are generating results from memformer
                if len(logits) == 2:
                    logits, memory = logits
                else:
                    logits, _, memory = logits
            logits = logits[:, -1, :]

            if greedy:
                sample = torch.argmax(logits)
            else:
                filtered_logits = top_k(logits, thres=filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)

            if eos_token is not None and (sample == eos_token).all():
                break

        out = out[:, t:]
        if num_dims == 1:
            out = out.squeeze(0)

        return out

    def forward(self, x, **kwargs):
        xi, xo = x[:, :-1], x[:, 1:]
        out = self.net(xi, **kwargs)
        loss = F.cross_entropy(out.transpose(1, 2), xo, ignore_index=self.ignore_index)
        return loss
