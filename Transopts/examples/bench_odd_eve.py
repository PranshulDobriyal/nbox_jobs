#!/usr/bin/env python3

# this is the singular unified script for benchmarking all the models and configurations
# in order to keep things simple and straight forward store all the required arguments in
# a seperate JSON object.

import os
import sys
import json
from argparse import ArgumentParser
import re
import numpy as np
import torch
from hashlib import md5
from tqdm import trange
from torch.nn import functional as F
import random
import nbox
from nbox.utils import folder, join, hash_
import itertools
# --- Add new modules: can we automate this process?

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# add the notebooks folder that has all the required code
sys.path.append("./notebooks")

from gpts import (
    get_xspan_model,
    get_fnets_model,
    get_gmlp_model,
    get_linformer_model,
    get_ftransformer_model,
    get_memformer_model,
    get_nystromformer_model,
    get_xcit_model,
    get_bart_model,
    get_gpt2_model,
    get_xlnet_model,
)
from gpts import TokenizerRuntimeEngine
from gpts import set_seed

# all the modules have a function called .get_model() and passing model arguments
GET_MODEL = {
    "expire_span": get_xspan_model,
    "fnet": get_fnets_model,
    "g_mlp_gpt": get_gmlp_model,
    "linformer": get_linformer_model,
    "feedback_transformer": get_ftransformer_model,
    "memformer": get_memformer_model,
    "nystromformer": get_nystromformer_model,
    "xcit": get_xcit_model,
    "bart": get_bart_model,
    "hf_gpt": get_gpt2_model,
    "xl_net": get_xlnet_model,

}


# --- modules loaded

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "adagrad": torch.optim.Adagrad,
    "adadelta": torch.optim.Adadelta,
    "adamax": torch.optim.Adamax,
}

# --- optim loaded

from common import set_seed, num_params
from autoregressive_wrapper import AutoregressiveWrapper

# --- loaded more modules from notebooks/ folder

GEN_SENT = "5 7 9 11 6 8"  # define any random string for the testing generation

# --- imports and loads complete


def get_tensors(seqlen=128, max=100):
    # Get the set of unique words
    all_seqs = []
    for i in range(1, max-seqlen*2):
        all_seqs.append([i+j*2 for j in range(seqlen)])
    
    # Create vocabulary and inverse vocabulary to convert words in to numbers and numbers to words respectively
    vocabulary = {str(i):i for i in range(1, 100)}
    mask = np.random.rand(len(all_seqs)) <= 0.9
    training_data = list(itertools.compress(all_seqs, mask))
    testing_data = list(itertools.compress(all_seqs, ~mask))
    input_ids = {"train": torch.from_numpy(np.array(training_data)), "test": torch.from_numpy(np.array(testing_data))}
    return input_ids, vocabulary

def hr(n=70):
    print("-" * n)


def json_load(path):
    # load any JSON like file with comment strings '//'
    # json files should have description strings so it's more friendly
    # but it won't load with json.load(f) so read the file, remove the
    # comments and json.loads(text)
    import json, re

    with open(path, "r") as f:
        text = f.read()
    text = re.sub(r"\s*(\/{2}.*)\n", "\n", text)
    config = json.loads(text)
    return config


def process_args(p):
    config = json_load(p)
    assert set(config.keys()) == {"models", "trainer"}, "Can have only two arguments `models` and `trainer`"
    trainer_args = config["trainer"]
    models = config["models"]
    return trainer_args, models


def forward(model, t, train=False):
    if train:
      model.train()
      logits = model(t[:, :-1])
    else:
      model.eval()
      with torch.no_grad():
        logits = model(t[:, :-1])
    if not isinstance(logits, torch.Tensor):
        logits = logits[0]
    B, S, V = logits.shape
    logits_flat = logits.view(-1, V)
    target_flat = t[:, 1:].contiguous().view(-1)
    loss = F.cross_entropy(logits_flat, target_flat)
    return logits, loss


def get_model(m, vocab_size):
    # function to get each model from configuration arguments
    if m["name"] not in GET_MODEL:
        raise ValueError(f"Model name: {m['name']} not found")

    kwargs = m["config"]
    kwargs["vocab_size"] = vocab_size

    m = GET_MODEL[m["name"]]
    model = m(**kwargs)[0]
    try:
        fn = m.forward
    except:
        fn = forward
    return model, fn


def train_model(model, data, train_args, forward_fn, model_name, device):
    model.train()
    data["train"] = data["train"].to(device)
    data["test"] = data["test"].to(device)
    model.to(device)
    optim = OPTIMIZERS[train_args["optim"]](model.parameters(), train_args["lr"])
    pbar = trange(train_args["n_steps"])
    acc_count = 0
    losses = []  # list of losses for each step

    for i in pbar:
        if i:
            pbar.set_description(f"Loss: {losses[-1]:.4f} | Model: {model_name}")
        logits, loss = forward_fn(model, data["train"], True)

        optim.zero_grad()  # removes previous looks gradient buffers
        loss.backward()  # fill gradient buffers
        optim.step()  # buffer -> update weights
        losses.append(loss.item())
        
        if len(losses)%5 == 0:
            seq = random.choice(data["test"])
            input_seq = seq[:-4]
            m = AutoregressiveWrapper(model)
            out = m.generate(input_seq, 4)

            if str(out) == str(seq[4:]):
                acc_count+=1
            else:
                acc_count =0

            if len(losses)%100 == 0:
                hr()
                print(" Input: ", input_seq)
                print(" Output: ", out)
                print("Acc counter: ", acc_count)
                hr()
            if acc_count>10:
                breakcd Trans
    
    with torch.no_grad():
        logits, loss = forward_fn(model, data["test"], False)
        hr()
        print("Validation Loss : ", loss.item())
            
    return losses


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--path", "-f", type=str, default="../configurations/full.jsonc", help="Path to configuration JSON")
    args.add_argument("--output-folder", "-o", type=str, default="./results", help="folder to store output JSON")
    args.add_argument("--n_generations", "-n", type=int, default=10, help="Number of generations")
    args = args.parse_args()

    # get the configurations
    trainer_args, models_args = process_args(args.path)

    os.makedirs(args.output_folder, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids, vocabulary = get_tensors(seqlen=8)
    t = input_ids
    inv_vocab = {v: k for k, v in vocabulary.items()}
    gen_sent = torch.tensor([vocabulary[x] for x in list(GEN_SENT.split(" "))]).long()
    gen_sent = gen_sent.to(device)
    all_losses = {}


    for m in models_args:
        # get the model, train it and get the losses over time
        set_seed(14)
        model, fn = get_model(m, len(vocabulary))
        losses = train_model(model, t, trainer_args, fn, m["name"], device)

        # create the AutoregressiveWrapper and then generate a few responses
        hr()
        print(f"Total Parameters: {num_params(model)}")
        print(f"Generating samples for {m['name']}")
        hr(50)
        model = AutoregressiveWrapper(model)
        generations = []
        for i in range(args.n_generations):
            out = model.generate(gen_sent, 25)
            sent = "".join([inv_vocab[x]+ " " for x in out.tolist()])
            if sent not in generations:
                print(sent)
                generations.append(sent)
            hr()

        # free the buffers
        del model

        all_losses[m["name"]] = {"losses": losses, "args": m}

    jout = json.dumps(all_losses)
    _hash = md5(jout.encode("utf-8")).hexdigest()
    sf = os.path.join(args.output_folder, f"{_hash}.json")
    print(">> Saving file to", sf)
    with open(sf, "w") as f:
        json.dump(all_losses, f)
