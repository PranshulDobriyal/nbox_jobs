from .expire_span import ExpireSpanGPT
from .feedback_transformer_pytorch import FeedbackTransformer
from .fnets import FourierGPT
from .g_mlp_gpt import gMLPGPT
from .linformer import LinformerLM
from .memformer import Memformer
from .nystromformer import Nystromformer
from .xcit import CrossCovarianceTestTransformer


def get_xspan_model(
    dim=16, depth=3, heads=2, seq_len=128, max_mem_len=32, ramp_len=16, vocab_size=39, dropout=0.0, expire_loss_coef=1e-6, **kwargs
):
    return ExpireSpanGPT(
        dim=dim,
        depth=depth,
        heads=heads,
        seq_len=seq_len,
        max_mem_len=max_mem_len,
        ramp_len=ramp_len,
        vocab_size=vocab_size,
        dropout=dropout,
        expire_loss_coef=expire_loss_coef,
    ), False


def get_ftransformer_model(**kwargs):
    config_dict = dict(
      num_tokens = 50257,           # number of tokens
      dim = 16,                     # dimension
      depth = 2,                    # depth
      seq_len = 128,                # the sequence length of each segment or window
      mem_len = 256,                # length of the memory buffer
      dim_head = 64,                # dimension of each head
      n_head = 8,                   # number of heads
      attn_dropout = 0.0,           # attention dropout
      ff_dropout = 0.0              # feedforward dropout
    )
    config_dict.update(kwargs)
    return FeedbackTransformer(
        **config_dict
    ), False


def get_fnets_model(dim=16, depth=3, seq_len=128, vocab_size=39, dropout=0.0, **kwargs):
    return FourierGPT(dim, depth, seq_len, vocab_size, dropout), False


def get_gmlp_model(
    dim, depth, vocab_size=39, seq_len=128, n_head=1, ff_mult=4, prob_survival=1.0, reversible=False, window=None, attn_dim=None, **kwargs
):
    return gMLPGPT(dim=dim, depth=depth, vocab_size=vocab_size, seq_len=seq_len, n_head=n_head,\
         ff_mult=ff_mult, prob_survival=prob_survival, reversible=reversible, window=window, attn_dim=attn_dim), False


def get_linformer_model(
    vocab_size,
    dim,
    seq_len,
    depth,
    k=256,
    heads=8,
    dim_head=None,
    one_kv_head=False,
    share_kv=False,
    reversible=False,
    dropout=0.0,
    **kwargs,
):
    return LinformerLM(
        num_tokens=vocab_size,
        dim=dim,
        seq_len=seq_len,
        depth=depth,
        k=k,
        heads=heads,
        dim_head=dim_head,
        one_kv_head=one_kv_head,
        share_kv=share_kv,
        reversible=reversible,
        dropout=dropout,
    ), False


def get_memformer_model(
    dim, num_memory_slots, vocab_size, max_seq_len=1024, heads=8, depth=3, num_mem_updates=1, mem_update_attn_heads=8, **kwargs
):
    return Memformer(
        dim=dim,
        num_memory_slots=num_memory_slots,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        heads=heads,
        depth=depth,
        num_mem_updates=num_mem_updates,
        mem_update_attn_heads=mem_update_attn_heads,
    ), False


def get_nystromformer_model(
    dim,
    depth,
    seq_len=128,
    dim_head=64,
    heads=8,
    num_landmarks=256,
    pinv_iterations=6,
    attn_values_residual=True,
    attn_values_residual_conv_kernel=33,
    attn_dropout=0.0,
    ff_dropout=0.0,
    vocab_size=0,
    lin_proj=False,
    in_feat=None,
    **kwargs,
):
    return Nystromformer(
        dim=dim,
        depth=depth,
        seq_len=seq_len,
        dim_head=dim_head,
        heads=heads,
        num_landmarks=num_landmarks,
        pinv_iterations=pinv_iterations,
        attn_values_residual=attn_values_residual,
        attn_values_residual_conv_kernel=attn_values_residual_conv_kernel,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        vocab_size=vocab_size,
        lin_proj=lin_proj,
        in_feat=in_feat,
    ), False


def get_xcit_model(dim=16, depth=3, n_head=4, seq_len=128, vocab_size=39, dropout=0.0, **kwargs):
    return CrossCovarianceTestTransformer(dim=dim, depth=depth, n_head=n_head, seq_len=seq_len, vocab_size=vocab_size, dropout=dropout), False
