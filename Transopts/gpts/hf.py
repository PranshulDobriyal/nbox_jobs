from transformers import BartForConditionalGeneration, BartConfig
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import XLNetConfig, XLNetLMHeadModel


def get_bart_model(
    vocab_size,
    d_model=16,
    encoder_layers=1,
    encoder_attention_heads=1,
    encoder_ffn_dim=64,
    decoder_ffn_dim=64,
    decoder_layers=1,
    decoder_attention_heads=1,
    max_position_embeddings=128,
    **kwargs,
):
    # Set the configuration for Bart
    config_dict = dict(
        vocab_size=vocab_size,
        d_model=d_model,
        encoder_layers=encoder_layers,
        encoder_attention_heads=encoder_attention_heads,
        encoder_ffn_dim=encoder_ffn_dim,
        decoder_ffn_dim=decoder_ffn_dim,
        decoder_layers=decoder_layers,
        decoder_attention_heads=decoder_attention_heads,
        max_position_embeddings=max_position_embeddings,
    )
    config_dict.update(kwargs)
    config = BartConfig(**config_dict)

    # Instantiate the model
    model = BartForConditionalGeneration(config)
    return model, True


def get_gpt2_model(vocab_size, d_model=16, n_layer=3, n_head=1, d_inner=64, **kwargs):
    # Set the configuration XLNet
    config = dict(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layer=n_layer,
        n_head=n_head,
        d_inner=d_inner,
    )
    config.update(kwargs)
    config = GPT2Config(**config)
    # Instantiate the model
    model = GPT2LMHeadModel(config)
    return model, True


def get_xlnet_model(vocab_size, d_model=16, n_layer=3, n_head=1, d_inner=64, **kwargs):
    # Set the configuration XLNet
    config = dict(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layer=n_layer,
        n_head=n_head,
        d_inner=d_inner,
    )
    config.update(kwargs)
    config = XLNetConfig(**config)
    # Instantiate the model
    model = XLNetLMHeadModel(config)
    return model, True


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    import numpy as np

    from einops import rearrange
    from tqdm import trange
    import matplotlib.pyplot as plt
    from newspaper import Article
    import re

    import os, sys

    currentdir = os.path.dirname(os.path.realpath(__file__))
    parentdir = os.path.dirname(currentdir)
    sys.path.append(parentdir)
    from common import num_params, set_seed

    def get_ds(text):
        # Get the set of unique words
        unq_words = sorted(list(set(text)))
        print("Vocab size:", len(unq_words))

        # Create vocabulary and inverse vocabulary to convert words in to numbers and numbers to words respectively
        vocabulary = {k: i for i, k in enumerate(unq_words)}
        inv_vocab = {i: k for i, k in enumerate(unq_words)}

        # create dataset

        # Set sequence size, this is the number of words that are sent to the model at once
        seq_size = 128

        # Create buckets, basically blocks of length = sequence_size. We remove the last element of the bucket to ensure constant sizes
        buckets = [text[i : i + seq_size] for i in range(0, len(text), seq_size)][:-1]

        # [abcde fgh] x 208 samples
        input_ids = np.array([[vocabulary[token] for token in sequence] for sequence in buckets])
        t = torch.from_numpy(input_ids)

        return t, vocabulary, inv_vocab

    def train_model(model, t, optim, n_steps=1000):
        all_losses = []
        pbar = trange(n_steps)
        for i in pbar:
            if i:
                pbar.set_description(f"Loss: {all_losses[-1]:.3f}")

            logits, loss = forward(model, t, True)
            optim.zero_grad()  # removes previous looks gradient buffers
            loss.backward()  # fill gradient buffers
            optim.step()  # buffer -> update weights
            all_losses.append(loss.item())
        return all_losses

    def get_text():
        def get_from_url(url):
            article = Article(url)
            article.download()
            article.parse()
            return article.text

        urls = [
            "https://towardsdatascience.com/lucy-says-hi-2031-agi-and-the-future-of-a-i-28b1e7b373f6",
            "https://towardsdatascience.com/to-do-great-data-science-embrace-domain-knowledge-167cb83dc050",
            "https://towardsdatascience.com/geometric-foundations-of-deep-learning-94cdd45b451d",
            "https://kitchingroup.cheme.cmu.edu/blog/2013/01/31/Smooth-transitions-between-discontinuous-functions/",
        ]

        text = "\n\n".join([get_from_url(u) for u in urls])
        text = re.sub(r"[^a-z0-9\s\.]", "", text.lower())
        return text

    set_seed(123)
    # Get the text
    text = get_text()
    t, vocab, inv_vocab = get_ds(text)

    # Get the model
    model = get_model(vocab_size=len(vocab))
    print("Number of parameters = ", model.num_parameters())

    # train
    n_steps = 1000
    optim = torch.optim.Adam(model.parameters())
    losses = train_model(model, t, optim, n_steps)

    # Generate and Save the plot for loss
    plt.plot(losses)
    plt.savefig("loss_curve.jpg")
