#!/usr/bin/env python3

# this is the singular unified script for benchmarking all the models and configurations
# in order to keep things simple and straight forward store all the required arguments in
# a seperate JSON object.

from importlib.resources import path
import os
import sys
import json
import torch
from hashlib import md5
from tqdm import trange
from argparse import ArgumentParser
from torch.nn import functional as F
from nbox import Operator
import nbox
from nbox.utils import folder, join, hash_
from gpts import TokenizerRuntimeEngine
from gpts import set_seed
from common import get_tensors, set_seed, get_text, num_params
from autoregressive_wrapper import AutoregressiveWrapper
# --- Add new modules: can we automate this process?

# add the notebooks folder that has all the required code
sys.path.append("./notebooks")
GEN_SENT = "i am a gpt."  # define any random string for the testing generation

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

def hr(n=70):
    print("-" * n)


class LoadData(Operator):
    def __init__(self) -> None:
        super().__init__()
        self.models = {
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
        self.optimizers = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "adagrad": torch.optim.Adagrad,
            "adadelta": torch.optim.Adadelta,
            "adamax": torch.optim.Adamax,
        }    
    def json_load(self, path):
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

    def forward(self, path):
        config = self.json_load(path)
        assert set(config.keys()) == {"models", "trainer"}, "Can have only two arguments `models` and `trainer`"
        trainer_args = config["trainer"]
        models = config["models"]
        config = [trainer_args, models]

        return self.models, self.optimizers, config

class TrainModel(Operator):
    def __init__(self, optim, inv_vocab, gen_sent, get_model, get_optim) -> None:
        super().__init__()
        self.GET_MODEL = get_model
        self.OPTIMIZER = optim
        self.inv_vocab = inv_vocab
        self.gen_sent = gen_sent
        self.OPTIMIZERS = get_optim


    def forwardpass(self, model, t, train=False):
        if train:
            model.train()
            logits = model(t[:, :-1])
            if not isinstance(logits, torch.Tensor):
                logits = logits[0]
            B, S, V = logits.shape
            logits_flat = logits.view(-1, V)
            target_flat = t[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(logits_flat, target_flat)
            return logits, loss
        else:
            model.eval()
            logits = model(t)
            return logits

    def get_model(self, m):
        # function to get each model from configuration arguments
        if m["name"] not in self.GET_MODEL:
            raise ValueError(f"Model name: {m['name']} not found")

        kwargs = m["config"]
        kwargs["vocab_size"] = 39

        m = self.GET_MODEL[m["name"]]
        model = m(**kwargs)[0]
        try:
            fn = m.forward
        except:
            fn = self.forwardpass
        return model, fn

    def train_model(self, model, data, train_args, forward_fn, model_name, device):
        data = data.to(device)
        model.to(device)
        optim = self.OPTIMIZERS[train_args["optim"]](model.parameters(), train_args["lr"])
        pbar = trange(train_args["n_steps"])
        losses = []  # list of losses for each step
        for i in pbar:
            if i:
                pbar.set_description(f"Loss: {losses[-1]:.4f} | Model: {model_name}")
            logits, loss = forward_fn(model, data, True)

            optim.zero_grad()  # removes previous looks gradient buffers
            loss.backward()  # fill gradient buffers
            optim.step()  # buffer -> update weights

            losses.append(loss.item())
        return losses

    def forward(self, m, t, trainer_args, device, n_gen=10):
        model, fn = self.get_model(m)
        losses = self.train_model(model, t, trainer_args, fn, m["name"], device)
       
        hr()
        print(f"Total Parameters: {num_params(model)}")
        print(f"Generating samples for {m['name']}")
        hr(50)
        model = AutoregressiveWrapper(model)
        for i in range(n_gen):
            out = model.generate(self.gen_sent, 25)
            sent = "".join([self.inv_vocab[x] for x in out.tolist()])
            print(sent)
            hr()

        # free the buffers
        del model

        return losses    

class TrainModels(Operator):
    def __init__(self, args: dict) -> None:
        super().__init__()
        data_loader = LoadData()
        self.GET_MODEL, self.OPTIMIZERS, config_data = data_loader(args["path"])
        self.trainer_args = config_data[0]
        self.models_args = config_data[1]
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.t, self.vocabulary = get_tensors(get_text())
        self.inv_vocab = {v: k for k, v in self.vocabulary.items()}
        GEN_SENT = "i am a gpt."  # define any random string for the testing generation
        gen_sent = torch.tensor([self.vocabulary[x] for x in GEN_SENT]).long()
        self.gen_sent = gen_sent.to(self.device)
        self.model_trainer = TrainModel(self.OPTIMIZERS, self.inv_vocab, self.gen_sent, self.GET_MODEL, self.OPTIMIZERS)

    def forward(self):
        os.makedirs(self.args["output_folder"], exist_ok=True)
        all_losses = {}

        for m in self.models_args:
            # get the model, train it and get the losses over time
            set_seed(14)
            losses = self.model_trainer(m, self.t, self.trainer_args, self.device)
            all_losses[m["name"]] = {"losses": losses, "args": m}

        jout = json.dumps(all_losses)
        _hash = md5(jout.encode("utf-8")).hexdigest()
        sf = os.path.join(self.args["output_folder"], f"{_hash}.json")
        print(">> Saving file to", sf)
        with open(sf, "w") as f:
            json.dump(all_losses, f)

#Load Data unify
