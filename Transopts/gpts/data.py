
import torch
import numpy as np
from typing import Dict, Union
from collections import Counter

class TokenizerRuntimeEngine():
  def __init__(self, fps, tokenizer, seqlen=512, batch_size=1):
    """TokenizerRuntimeEngine takes in list of files along with it's meta data and becomes a callable generator.
    When calling you can tell it what kind of data that you want. It is a full fledged data engine in itself.
    This will sit in nbox one day and thus has to be engineered in such a what that it is production grade with
    good documentation. In the nbox hierarchy it sits parallel to nbox.Model thus has to continue the following
    traits:
    1) __call__ with the input instructions
    2) can take in same kind of instructions as nbox.Parsers i.e.:
      * primitive that tells the actual fetching instruction
      * structure should be same as the source meta data
    
    Args:
      fps (list): The file paths have to be the primary index inside the lists and so filepaths "fps" can look like these:
          1) list of strings: ["file1.txt", "file2.txt", ...]
          2) list of dicts: [{"file1.txt": "cat1"}, {"file2.txt": "cat2"}, ...]
          3) dict of strings: {"file1.txt": "cat1", "file2.txt": "cat2", ...}
          4) dict of categories: {"cat1": ["file1.txt", "file2.txt", ...], "cat2": ["file3.txt", "file4.txt", ...]}
        
      tokenizer (TokenizerObject): Object of the Tokenizer
      seqlen (int, optional): Length of the Sequence. Defaults to 512.
      batch_size (int, optional): Size of the Batches. Defaults to 1.
    """
    self.tokenizer = tokenizer
    self.seqlen = seqlen
    self.batch_size = batch_size

    # parse the fps and covert to format
    # {"meta": ["file1.txt", "file2.txt", ...]}
    if isinstance(fps, list):
      if isinstance(fps[0], str):
        # print("fps - list - str")
        fps = {"null": fps}  # list of files will start with null category
      elif isinstance(fps[0], dict):
        # print("fps - list - dict")
        fps = {}
        for x in fps:
          k = list(x.keys())[0]
          v = list(x.values())[0]
          fps.setdefault(v, []).append(k)  # list of dicts will start with category as key
      else:
        # print(f"fps - list - {type(fps[0])}")
        raise ValueError("fps is not in the correct format")
    elif isinstance(fps, dict):
      k = next(iter(fps))
      v = fps[k]
      assert isinstance(k, str), f"key has to be a string got: {type(k)}"
      if isinstance(v, list):
        # dict with list as values is the format we want, so just check and go
        # print("fps - dict - list")
        assert all([isinstance(_v, list) for _k,_v in fps.items()]), "All values should be a list"
      elif isinstance(v, str):
        # print("fps - dict - str")
        assert all([isinstance(_v, str) for _k,_v in fps.items()]), "All values should be a string"
        fps[k] = [v]  # dict with strings as values gets converted to list of strings
    else:
      raise ValueError(f"fps is not in the correct format: {type(fps)}")
    self.fps = fps

    # Open the first file in all the metas and create tokens buffer
    self._f = {k: open(v[0], "r") for k,v in self.fps.items()}                #classes ki first file khol ke baith gaya
    self.tokens_buffer = {k: [] for k,v in self.fps.items()}
    self.curr_f_idx = {k: 0 for k,v in self.fps.items()}

    self.__device = "cpu"

  def _read_chunk(self, f_key, size=1024):
    while True:
      b = self._f[f_key].read(size)
      if not b:
        break
      else:
        yield b

  def _sample(self, size):
    n_open_files = len(self._f)
    samples_ = np.random.choice(np.repeat(np.arange(n_open_files), size),size=size, replace=False).tolist()
    counts_per_file = Counter(samples_)
    file_to_count_map = {f:c for f,(_,c) in zip(self._f, counts_per_file.items())}
    return file_to_count_map

  def to(self, device_map: torch.device):
    self.__device = device_map

  def get_input_ids(self, f_key, n, seqlen):
    f = self._f[f_key]
    input_ids = []
    label_ids = []
    for i in range(n):
      while len(self.tokens_buffer[f_key]) < seqlen:
        chars = []
        # Check if any file is open
        if type(self._f[f_key]) is not int  :
            chars = next(self._read_chunk(f_key, seqlen * 10))
            toks = self.tokenizer(chars)["input_ids"]
            self.tokens_buffer[f_key].extend(toks)
        else:
            # If no file is open, that means we have read all files. We start reading from the first file again.
            self._f[f_key] = open(self.fps[f_key][0], "r")
            continue

        if len(chars) < seqlen * 10:
            # since the chunk we got is smaller than the required seqlen the file has clearly ended
            # so we close the current file and delete the file object
            self._f[f_key].close()
            self._f[f_key] = -1
            self.curr_f_idx[f_key] += 1
            if self.curr_f_idx[f_key] >= len(self.fps[f_key]):
              #If all files from the current file have been read, add the padding tokens, reset the curr_f_idx
                pad_tokens = [50256]
                self.tokens_buffer[f_key].extend(pad_tokens)
                self.curr_f_idx[f_key] = 0
                break
            else:
              # more files are left to be read. Open the next file if True
              self._f[f_key] = open(self.fps[f_key][self.curr_f_idx[f_key]], "r", encoding="utf-8", errors="ignore")              

      # Extract and Pad the input_ids and labels
      if len(self.tokens_buffer[f_key]) < seqlen:
          label_buffer = self.tokens_buffer[f_key] + [-100 for _ in range(seqlen - len(self.tokens_buffer[f_key]))]
          input_buffer = self.tokens_buffer[f_key] + [50256 for _ in range(seqlen - len(self.tokens_buffer[f_key]))]
      else:
          label_buffer = self.tokens_buffer[f_key][:seqlen]
          input_buffer = label_buffer
      input_ids.append(input_buffer)
      label_ids.append(label_buffer)

      # Delete the tokens that are being returned from the tokens_buffer
      del self.tokens_buffer[f_key][:seqlen]

    input_ids = torch.tensor(input_ids)
    label_ids = torch.tensor(label_ids)

    return {
      "input_ids": input_ids,
      "labels": label_ids
    }

  def __getitem__(self, batch_meta: Union[Dict[str, int], int] = None, seqlen: int = None):
    """Get the sampled data, just the way you want it

    Args:
        batch_meta (Union[Dict[str, int], int], optional): what should be the batch composition,
          if None then batch_size data is returned by sampling set of open files.
        seqlen (int, optional): what should be the sequence length. Defaults to None.

    Returns:
        dict: {"input_ids": Tensor, "labels": Tensor}
    """
    # batch_meta brought to a standard structure like {"cat1": 13, "cat2": 5, ...}
    if batch_meta is None or isinstance(batch_meta, int):
      # get sample as much as batch size
      batch_meta = self._sample(self.batch_size if batch_meta is None else batch_meta)
    elif isinstance(batch_meta, dict):
      assert isinstance(list(batch_meta.values())[0], int), "batch_meta should be a dict of ints"
      # check if keys in batch_meta are also in _f
      open_metas = set(self._f.keys())
      batch_metas = set(batch_meta.keys())
      if not batch_metas.issubset(open_metas):
        raise KeyError(batch_metas.difference(open_metas))
    else:
      raise KeyError(batch_meta)

    seqlen = seqlen if seqlen is not None else self.seqlen

    # now read the files based on batch_meta
    # Q: can this be multi-threaded or multiprocessed to make it faster?
    input_ids = []
    labels = []
    for k,v in batch_meta.items():
      out = self.get_input_ids(k, v, seqlen)
      input_ids.append(out["input_ids"])
      labels.append(out["labels"])

    input_ids = torch.cat(input_ids, dim=0)
    labels = torch.cat(labels, dim=0)

    if self.__device != "cpu":
      input_ids = input_ids.pin_memory().to(self.__device)
      labels = labels.pin_memory().to(self.__device)

    return {"input_ids": input_ids, "labels": labels}


  def num_rows(self, x):
    dim = x.size()
    if len(dim) == 1:
      return 1
    else:
      return dim[0]

  def reset_file_idx(self):
    self.curr_f_idx = 0
    self._f = self._f = open(self.fps[self.curr_f_idx], "r", encoding="utf-8", errors="ignore")


  def __call__(self, idx=None, seqlen=None, batch_size=None):
    """Read chunks till seqlen and name it input_ids
    margin of 10, ie. we assume that 10 chars make one token, it is actully 6.383727639930756
    Returns a batch of size = self.batch_size

    Args:
      idx ([int]): This variable is not used but since this is a built in function we have to include it

    Returns:
      Dict : {"input_ids": tensor, "labels": tensor}
    """
    batch_size = self.batch_size if batch_size is None else batch_size
    # Get the first set of Input Ids and Labels
    try:
      item = self.get_input_ids(seqlen=seqlen)
    except StopIteration:
      # If no file is open, set curr_f_idx to 0 and open the first file
      self.reset_file_idx()
      item = self.get_input_ids(seqlen=seqlen)

    # These variables store the final tensors to be returned
    batch_input_ids = item["input_ids"]
    batch_labels = item["labels"]

    # Run the loop until the number of rows of the batch_input_ids tensor is less than batch_size
    while self.num_rows(batch_input_ids) < batch_size:
      try:
        item = self.get_input_ids(seqlen=seqlen)
        batch_input_ids = torch.vstack((batch_input_ids, item["input_ids"].unsqueeze(0)))
        batch_labels = torch.vstack((batch_labels, item["labels"].unsqueeze(0)))
      except StopIteration:
        # If no file is open, set curr_f_idx to 0 and open the first file
        self.reset_file_idx()

    return {"input_ids": batch_input_ids, "labels": batch_labels}

  def __iter__(self):
    """Iterator Function

    Yields:
      dict: {"input_ids": Tensor, "labels": Tensor}
    """
    yield self.__call__()
