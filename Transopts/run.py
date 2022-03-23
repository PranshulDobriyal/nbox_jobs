#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gpts
from fire import Fire
from transformers.models.auto.tokenization_auto import AutoTokenizer

from gpts import *

# ---------
import nbox
# ---------

def main(
  arch: str,
  data_train: str,
  data_test: str,
  tokenizer: str,
):
  """train GPT objective model with by giving the training and testing data.

  Args:
      arch (str): Architecture of the underlying DAG
      data_train (str): Path to the training data
      data_test (str): Path to the validation data
      tokenizer (str): tokenizer to use hf.co/tokenizers
  """
  assert hasattr(gpts, f"get_{arch}_model"), f"Cannot find model builder for: {arch}"

  t = AutoTokenizer.from_pretrained(tokenizer)


if __name__ == '__main__':
  Fire(main)
