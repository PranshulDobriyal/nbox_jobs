import os
import re
import torch
import numpy as np
import random
from newspaper import Article

from nbox.utils import folder, join


def get_text():
    # To get text from the webpage(s)
    def get_text_from_url(url):
        article = Article(url)
        article.download()
        article.parse()
        return article.text

    fp = join(folder(__file__), "text.txt")
    if not os.path.exists(fp):
        print("Downloading text...")

        urls = [
            "https://towardsdatascience.com/lucy-says-hi-2031-agi-and-the-future-of-a-i-28b1e7b373f6",
            "https://towardsdatascience.com/to-do-great-data-science-embrace-domain-knowledge-167cb83dc050",
            "https://towardsdatascience.com/geometric-foundations-of-deep-learning-94cdd45b451d",
            "https://kitchingroup.cheme.cmu.edu/blog/2013/01/31/Smooth-transitions-between-discontinuous-functions/",
        ]
        text = "\n\n".join([get_text_from_url(u) for u in urls])

        # This Regex commands strips the texts so only alphabets, numbers, spaces and dot (.) remain
        text = re.sub(r"[^a-z0-9\s\.]", "", text.lower())

        with open(fp, "w") as f:
            f.write(text)

    else:
        with open(fp, "r") as f:
            text = f.read()

    return text


def get_tensors(text):
    # Get the set of unique words
    unq_words = sorted(list(set(text)))

    # Create vocabulary and inverse vocabulary to convert words in to numbers and numbers to words respectively
    vocabulary = {k: i for i, k in enumerate(unq_words)}

    # create dataset

    # Set sequence size, this is the number of words that are sent to the model at once
    seq_size = 128

    # Create buckets, basically blocks of length = sequence_size. We remove the last element of the bucket to ensure constant sizes
    buckets = [text[i : i + seq_size] for i in range(0, len(text), seq_size)][:-1]

    # [abcde fgh] x 208 samples
    # [123450678]
    input_ids = np.array([[vocabulary[token] for token in sequence] for sequence in buckets])
    t = torch.from_numpy(input_ids)
    return t, vocabulary


def num_params(model):
    return sum(p.numel() for p in model.parameters())


# Setting seed so that the random results generated can be reproduced
def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
