#!/usr/bin/env python

import torch
from torch import nn
import torch.nn.functional as F

from tqdm.auto import tqdm

# ---------
import nbox

# ---------


class Core(nn.Module):
  def __init__(self, model):
    super().__init__()
    self.model = model
    self._is_hf_model = True

  def eval(self):
    self.model.eval()

  def train(self):
    self.model.train()

  def _log(self):
    pass

  def train_once(
    self,
    ds_train,
    ds_test,
    n_step,
    opt,
    eval_every,
    gas=1,
  ):
    # this function takes in an iterator dataset object and trains the model
    # parameters are sent via kwargs

    pbar = tqdm(n_step)
    _ds = iter(ds_train)
    for i in pbar:
      try:
        x = next(_ds)
      except StopIteration:
        _ds = iter(ds_train)
        x = next(_ds)

      logger = {}

      # different types of calls based on model source
      if self._is_hf_model:
        out = self.model(x, labels=x)
        _loss = out.loss
      else:
        out = self.model(x)
        _loss = F.cross_entropy(out, x)

      _loss = _loss.mean() / gas
      _loss.backward()

      logger["loss"] = (_loss * gas).item()

      if i % gas == 0:
        opt.step()
        opt.zero_grad()

      pbar.set_description(f"loss: {_loss.item():.4f}")

      # if this at eval step, run all this again for eval
      if (i + 1) % eval_every == 0:
        self.eval()
        loss_evals = []
        _ds_ = iter(ds_test)
        for x in _ds_:
          if self._is_hf_model:
            out = self.model(x, labels=x)
            _loss = out.loss
          else:
            out = self.model(x)
            _loss = F.cross_entropy(out, x)
          loss_evals.append(_loss.item())
        self.train()
        logger["eval_loss"] = sum(loss_evals) / len(loss_evals)

      self._log(logger)

  def __call__(self, x):
    return self.model(x)
