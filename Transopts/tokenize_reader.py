import os
import re
import json
from torch._C import Value
import wandb
import random
import shutil
import numpy as np
import math
from glob import glob
from tqdm import trange
from types import SimpleNamespace
from argparse import ArgumentParser
from pprint import pprint as peepee
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, optimization, GPT2Tokenizer, AutoTokenizer
from notebooks.autoregressive_wrapper import AutoregressiveWrapper


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_avg_vocab_len(tokenizer_name):
    """Returns the average length of all the tokens"""
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except:
        raise ValueError("Invalid Tokenizer Name")
    return sum(list(map(len, tokenizer.vocab))) / len(tokenizer.vocab)


# --------- tokenized_reader --------- #


class TokenizerReaderRuntime(Dataset):
    def __init__(self, fps, tokenizer, seqlen=512, batch_size=1):
        """Tokenized Reader Class Constructor

        Args:
            fps (list): list containing the names of all the input files
            tokenizer (TokenizerObject): Object of the Tokenizer
            seqlen (int, optional): Length of the Sequence. Defaults to 512.
            batch_size (int, optional): Size of the Batches. Defaults to 1.
        """
        self.tokenizer = tokenizer
        self.fps = fps
        self.seqlen = seqlen
        self.seqlen = seqlen
        self.tokens_buffer = []  # this is a rolling list with all the input_ids
        self.curr_f_idx = 0  # contains the index of the file being read
        self.size = 0
        self.batch_size = batch_size
        # Open the first file
        self._f = open(fps[0], "r", encoding="utf-8", errors="ignore")

    def _read_chunk(self, size=1024):
        """Reads a chunk from the currently open file

        Args:
            size (int, optional): Size of the chunk to be read. Defaults to 1024.

        Yields:
            [Str]: [Chunk Read from the file]
        """
        while True:
            b = self._f.read(size)
            if not b:
                break
            else:
                yield b

    def num_rows(self, x):
        """Helper Function to retrieve the number of rows from a tensor.

        Args:
            x (Tensor): The Tensor whose number of rows are required

        Returns:
            int: Returns 1 if the tensor is 1-D, else returns the number of rows of the tensor
        """
        dim = x.size()
        if len(dim) == 1:
            return 1
        else:
            return dim[0]

    def reset_file_idx(self):
        """
        Helper Function to reset the curr_f_idx and open the first file
        """
        self.curr_f_idx = 0
        self._f = self._f = open(self.fps[self.curr_f_idx], "r", encoding="utf-8", errors="ignore")

    def get_input_ids(self, seqlen=None):
        """Reads from the file and returns input_ids and labels

        Args:
            seqlen (int, optional): No Use actually [Defaults to None]

        Raises:
            StopIteration: When all files have been read

        Returns:
            Dict: {"input_ids": [], "labels": []}
        """
        seqlen = seqlen if seqlen is not None else self.seqlen
        while len(self.tokens_buffer) < seqlen:
            chars = []
            # Check if any file is open
            if hasattr(self, "_f"):
                chars = next(self._read_chunk(seqlen * 10))
                toks = self.tokenizer(chars)["input_ids"]
                self.tokens_buffer.extend(toks)
            else:
                # If no file is open, that means we have read all files. Raise an error for the same.
                raise StopIteration("All files have been read")
            if len(chars) < seqlen * 10:
                # since the chunk we got is smaller than the required seqlen the file has clearly ended
                # so we close the current file and delete the file object
                self._f.close()
                del self._f
                self.curr_f_idx += 1
                # Check if more files are left to be read. Open the next file if True
                if self.curr_f_idx >= len(self.fps):
                    pad_tokens = [50256]
                    self.tokens_buffer.extend(pad_tokens)
                    break
                else:
                    self._f = open(self.fps[self.curr_f_idx], "r", encoding="utf-8", errors="ignore")

        # Extract and Pad the input_ids and labels
        if len(self.tokens_buffer) < seqlen:
            label_buffer = self.tokens_buffer + [-100 for _ in range(seqlen - len(self.tokens_buffer))]
            input_buffer = self.tokens_buffer + [50256 for _ in range(seqlen - len(self.tokens_buffer))]
        else:
            label_buffer = self.tokens_buffer[:seqlen]
            input_buffer = label_buffer
        input_ids = torch.tensor(input_buffer).long()
        label_ids = torch.tensor(label_buffer).long()

        # Delete the tokens that are being returned from the tokens_buffer
        del self.tokens_buffer[:seqlen]
        return {"input_ids": input_ids, "labels": label_ids}

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


class Data(Dataset):
    def __init__(self, file_dict, seqlen=128, batch_size=2, tokenizer=GPT2Tokenizer.from_pretrained("gpt2")):
        """Initialisation Function

        Args:
            file_dict (dict): Contains the meta map of file. Keys are the classes, values hold file names
            seqlen (int, optional):  Sequence Length. Defaults to 128.
            batch_size (int, optional): Batch Size. Defaults to 2.
            tokenizer (TokenizerObject, optional): Object of the tokenizer. Defaults to GPT2Tokenizer.from_pretrained("gpt2").
        """
        self.file_dict = file_dict
        self.seqlen = seqlen
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.token_read_class = {}
        self.data_buffers = {x: [] for x in self.file_dict.keys()}  # Dictionary holding objects of TokenizerReaderRuntime Class
        for key in self.file_dict.keys():
            self.token_read_class[key] = TokenizerReaderRuntime(
                fps=self.file_dict[key], tokenizer=self.tokenizer, seqlen=self.seqlen, batch_size=1
            )

    def num_rows(self, x):
        """Helper Function to retrieve the number of rows from a tensor.

        Args:
            x (Tensor): The Tensor whose number of rows are required

        Returns:
            int: Returns 1 if the tensor is 1-D, else returns the number of rows of the tensor
        """
        dim = x.size()
        if len(dim) == 1:
            return 1
        else:
            return dim[0]

    def __getitem__(self, items):
        """Returns Input IDs and Labels for given configuration

        Args:
            items (Dict/Set/None): contains one or two of file maps and seqlen

        Returns:
            dict: {"input_ids": Tensor, "labels": Tensor}
        """
        # If Nothing is passed
        num_reads = {x: math.ceil(self.batch_size / len(self.file_dict.keys())) for x in self.file_dict.keys()}
        seqlen = self.seqlen

        # If only file maps are passed
        if type(items) is dict:
            num_reads = {x: items[x] for x in self.file_dict.keys()}

        # if filemaps and seqlen are provided
        elif items is not None:
            num_reads = {x: items[0][x] for x in self.file_dict.keys()}
            seqlen = items[1]

        for x in num_reads.values():
            if type(x) is not int:
                raise ValueError("Number of reads can only be an integer")

        input_ids_buffer = torch.tensor([])
        labels_buffer = torch.tensor([])
        for key in num_reads.keys():
            data_buffer = self.token_read_class[key].__call__(None, seqlen=seqlen, batch_size=num_reads[key])
            input_ids_buffer = (
                torch.vstack((input_ids_buffer, data_buffer["input_ids"])) if len(input_ids_buffer > 0) else data_buffer["input_ids"]
            )
            labels_buffer = torch.vstack((labels_buffer, data_buffer["labels"])) if len(labels_buffer > 0) else data_buffer["labels"]

        return {"input_ids": input_ids_buffer, "labels": labels_buffer}

    def __iter__(self):
        yield self.__getitem__(None)


# ______________ Binary Reader ________________ #


class BinaryReader(Dataset):
    def __init__(self, fps, seqlen=2):
        """
        fps: all the filepaths to load
        seqlen: sequence length to use
        """
        # binarize EOT tag
        EOT = "<|endoftext|>".encode("utf-8")
        sections = []

        read_chunk_size = 5 * 10
        chunk_mult = (
            read_chunk_size // seqlen
        )  # How many sequences are there in a chunk we read. Ig we do this to read more data in a go and reduce I/O time
        self.sizes = []
        self.fps = fps
        self.seqlen = seqlen

        for fp in fps:
            # open file -> read chunks -> total length of chunks -> close file
            self._f = open(fp, "r", encoding="utf-8", errors="ignore")
            _ch_sz = sum([chunk_mult for bl in self._read_chunk(read_chunk_size)])
            print("-->", fp, _ch_sz)
            self.sizes.append(_ch_sz)
            self._f.close()
        print("Size of chunks", self.sizes)
        self.curr_f_idx = 0
        self._f = open(self.fps[self.curr_f_idx], "r", encoding="utf-8", errors="ignore")

    def _read_chunk(self, size=5):
        while True:
            b = self._f.read(size)
            # print("="*50, "\nb = ", b)
            if not b:
                break
            else:
                # print("yielding ",list(b.encode("utf-8")))
                yield list(b.encode("utf-8"))

    def __len__(self):
        # since we lazy read the files, we only return the number of chunks
        # in the dataset
        return sum(self.sizes)

    def __getitem__(self, *args, **kwargs):
        # since we lazy read the files, we don't keep track of indices thus it is not deterministic
        # is this good, probably not, but the idea is that with larger amount of raw data it won't matter
        input_ids = next(self._read_chunk(self.seqlen))
        print("Input Ids = ", input_ids)
        labels = None
        if len(input_ids) < self.seqlen:
            # since the chunk we got is smaller than the required seqlen the file has clearly ended
            # so we close the current file, delete the file object and open the next file
            self._f.close()
            del self._f

            self.curr_f_idx += 1
            self._f = open(self.fps[self.curr_f_idx], "r", encoding="utf-8", errors="ignore")

            # we'll pad the input sequence and labels are filled with -100 to ignore those values
            # in labels we are adding a special EOT ie. 256 (257th item in vocab)
            labels = [x for x in input_ids] + [256] + [-100 for _ in range(self.seqlen - len(input_ids) - 1)]
            input_ids += [256 for _ in range(self.seqlen - len(input_ids))]

        # convert to tensors and labels if given and return the dicitonary
        input_ids = torch.tensor(input_ids).long()
        labels = input_ids.clone() if labels == None else torch.tensor(labels).long()

        # I don't know but sometimes the chunks are incorrect
        input_ids = input_ids[: self.seqlen]
        labels = labels[: self.seqlen]
        return {"input_ids": input_ids, "labels": labels}


class TrainerConfig:
    def __init__(self, **kwargs):
        self.train_steps = int(1e7)
        self.gas = 1  # should be >= 1
        self.test_every = 100
        self.patience = 4
        self.keep_max = 5
        self.max_test_steps = 300
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(
        self,
        model,
        train_conf,
        train_dl,
        test_dl,
        optim,
        save_folder,
        lr_scheduler=None,
        wandb=False,
        test_sequences=["Hello World", "def add_two_numbers(a, b):"],
    ):
        self.model = model
        self.c = train_conf
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.optim = optim
        self.lr_scheduler = lr_scheduler

        self.device = "cpu"
        try:
            self.vocab_size = self.model.config.vocab_size
        except:
            self.vocab_size = len(AutoTokenizer.from_pretrained("gpt2"))
        if torch.cuda.is_available():
            print("Model is now CUDA!")
            self.device = torch.cuda.current_device()
            if torch.cuda.device_count() > 1:
                print("Model is now DataParallel!")
                self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.to(self.device)

        self.test_sequences = [[y for y in x.encode("utf-8")] for x in test_sequences]
        self.wandb = wandb
        self.device = next(iter(self.model.parameters())).device
        self.save_folder = save_folder

        self.save_files = []

    def load(self, save_folder):
        fp_m = save_folder + "/model.pt"
        fp_optim = save_folder + "/optim.pt"
        fp_lrs = save_folder + "/lrs.pt"

        self.model.load_state_dict(torch.load(fp_m, map_location="cpu"))
        if torch.cuda.is_available():
            print("Model is now CUDA!")
            self.device = torch.cuda.current_device()
            if torch.cuda.device_count() > 1:
                print("Model is now DataParallel!")
                self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.to(self.device)
        self.optim.load_state_dict(torch.load(fp_optim, map_location="cpu"))
        self.lr_scheduler.load_state_dict(torch.load(fp_lrs, map_location="cpu"))

    def save(self, idx):
        m = self.model
        c = self.c
        train_dl = self.train_dl
        test_dl = self.test_dl
        optim = self.optim
        lr_scheduler = self.lr_scheduler

        save_folder = self.save_folder + f"/{idx}"
        os.makedirs(save_folder, exist_ok=True)

        fp_m = save_folder + "/model.pt"
        fp_optim = save_folder + "/optim.pt"
        fp_lrs = save_folder + "/lrs.pt"

        print("Saving model at", fp_m)
        torch.save(m.state_dict(), fp_m)
        print("Saving optimizer at", fp_optim)
        torch.save(optim.state_dict(), fp_optim)
        if lr_scheduler != None:
            print("Saving lr_scheduler at", fp_lrs)
            torch.save(lr_scheduler.state_dict(), fp_lrs)

        self.save_files.append(save_folder)

        if len(self.save_files) > c.keep_max:
            to_remove_folder = self.save_files[-c.keep_max - 1]
            # shutil because os.rmdir does not remove filled directory
            print("Removing", to_remove_folder)
            shutil.rmtree(to_remove_folder)
            del self.save_files[-c.keep_max - 1]

    def __call__(self, x, train=False, optim=None, lr_scheduler=None, loss_scale=1.0):
        # ambitious move by making trainer callable, this performs one pass
        # (both forward and backward). when this is called from the .train()
        # method, it does not look like it is removing any complexity. But
        # infact this is creating new oppurtunities when it comes to experimenting
        # training steps. This sits directly one level above playing with the
        # gradients.
        m = self.model
        logger = {}
        x = {k: v.to(self.device) for k, v in x.items()}

        # different conditionals for inference and training
        if not train:
            # inference is priority
            m.eval()
            with torch.no_grad():
                try:
                    out = m(**x)
                    loss = out.loss.mean()
                except:
                    out = m(x["input_ids"])
                    if not isinstance(out, torch.Tensor):
                        out = out[0]
                    shifted_logits = out[..., :-1, :].contiguous()
                    shifted_labels = x["labels"][..., 1:].contiguous()
                    # Flatten the tokens
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
            return loss
        else:
            # just because this is training does not mean optimizer needs to be
            # provided. optimizer and lr_scheduler can be called whenever we want
            if optim != None:
                assert hasattr(optim, "step"), "Provide correct optimizer"
                if lr_scheduler != None:
                    assert hasattr(lr_scheduler, "step"), "Provide correct LR Scheduler"
            m.train()

            # forward pass
            try:
                out = m(**x)
            except:
                out = m(x["input_ids"])

            if not isinstance(out, torch.Tensor):
                out = out[0]

            try:
                loss = out.loss.mean()
            except:
                # Shift so that tokens < n predict n
                shifted_logits = out[..., :-1, :].contiguous()
                shifted_labels = x["labels"][..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
            loss = loss / loss_scale
            loss.backward()

            logger["train_loss"] = loss.item() * loss_scale

            if optim != None:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                optim.step()
                m.zero_grad()
                if lr_scheduler != None:
                    lr_scheduler.step()
                    logger["lr"] = lr_scheduler.get_last_lr()[0]

            return logger

    def train(self):
        m = self.model
        c = self.c
        train_dl = self.train_dl
        test_dl = self.test_dl
        optim = self.optim
        lr_scheduler = self.lr_scheduler

        iter_train_dl = iter(train_dl)
        best_test_loss = 1e7
        patience = 1

        all_train_losses = []
        all_test_losses = []

        pbar = trange(c.train_steps)
        for i in pbar:
            if i:
                desc_str = f"{all_train_losses[-1]:.3f}"
                pbar.set_description(desc_str)

            try:
                x = next(iter_train_dl)
                # x = train_dl[{"a": 3, "c": 7}]
            except StopIteration:
                train_dl = self.train_dl
                iter_train_dl = iter(train_dl)
                x = next(iter_train_dl)

            if (i + 1) % c.gas == 0:
                logger = self(x, True, optim=optim, lr_scheduler=lr_scheduler, loss_scale=c.gas)
            else:
                logger = self(x, True, loss_scale=c.gas)
            all_train_losses.append(logger["train_loss"])

            if (i + 1) % c.test_every == 0:
                test_losses = []
                iter_test_dl = iter(test_dl)
                for _, x in zip(range(c.max_test_steps), iter_test_dl):
                    # test_dl can be very big so we only test a few steps
                    _tl = self(x, False)
                    test_losses.append(_tl.item())
                test_loss = np.mean(test_losses)
                logger["test_loss"] = test_loss
                all_test_losses.append(logger["test_loss"])

                # generate a few sequences
                all_s = []
                gen_table = wandb.Table(columns=["sequences"])
                for s in self.test_sequences:
                    _m = self.model.module if hasattr(self.model, "module") else self.model
                    try:
                        out = _m.generate(
                            input_ids=torch.tensor(s).unsqueeze(0).to(self.device),
                            max_length=128,
                            pad_token_id=self.vocab_size - 1,
                            temperature=0.95,
                            top_p=1.0,
                            num_return_sequences=5,
                            do_sample=True,
                        )
                    except:
                        out = AutoregressiveWrapper(_m).generate(torch.tensor(s).unsqueeze(0).to(self.device), 25)
                    samples = tokenizer.batch_decode(out)  # [str(bytes(x.cpu().tolist()))[2:-1] for x in out]
                    all_s.extend(samples)
                for x in samples:
                    gen_table.add_data(x)
                for x in all_s:
                    print("-->", x)
                logger["sequences"] = gen_table

                # save the model if required
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    self.save(i)
                    print("=========================\n", i)

                    patience = 0
                else:
                    patience += 1

            # # log the data
            # if self.wandb:
            #   wandb.log(logger)

            # running out of patience much?
            if patience == c.patience:
                print("Break Training")
                break


# Cuda testing
# model arch = Models in Transopts
# Add this to transopts repo
# finally test it and close


def get_model(args, tokenizer):
    # --- Add new modules: can we automate this process?

    # add the notebooks folder that has all the required code
    import sys

    sys.path.append("./notebooks")

    import base_hf_gpt
    from bart import bart
    from xl_net import xl_net
    from g_mlp_gpt import g_mlp_gpt
    from linformer import linformer
    from nystromformer import nystromformer
    from feedback_transformer import feedback_transformer
    from expire_span import model as expire_span_model
    from xcit import model as xcit_model
    from linformer import linformer
    from memformer import memformer
    from fnets import fnet as fnet_model
    from common import num_params

    # all the modules have a function called .get_model() and passing model arguments

    MODEL_TO_MODULE = {
        "hf_gpt": base_hf_gpt,
        "g_mlp_gpt": g_mlp_gpt,
        "bart": bart,
        "xl_net": xl_net,
        "linformer": linformer,
        "nystromformer": nystromformer,
        "feedback_transformer": feedback_transformer,
        "expire_span": expire_span_model,
        "xcit": xcit_model,
        "memformer": memformer,
        "fnet": fnet_model,
    }

    model = None
    # define the model
    # using if else ladder as we want to import only the model we need
    if args.model_arch == "gpt2":
        model_config = GPT2Config(
            vocab_size=tokenizer.vocab_size + 1,
            n_positions=args.seqlen,
            n_ctx=args.seqlen,
            n_embd=args.dim,
            n_layer=args.n_layer,
            n_head=args.n_head,
        )
        model = GPT2LMHeadModel(model_config)

    elif args.model_arch == "bart":
        from notebooks.bart import bart

        config_dict = dict(
            d_model=args.dim,
            encoder_layers=args.n_layer,
            encoder_attention_heads=args.n_head,
            encoder_ffn_dim=args.dim * 4,
            decoder_ffn_dim=args.dim * 4,
            decoder_layers=args.n_layer,
            decoder_attention_heads=args.n_head,
            max_position_embeddings=args.seqlen,
            vocab_size=tokenizer.vocab_size,
        )
        _model = bart

    elif args.model_arch == "memformer":
        from notebooks.memformer import memformer

        config_dict = dict(
            dim=args.dim,
            depth=args.n_layer,
            n_head=args.n_head,
            num_memory_slots=args.dim * 2,
            mem_update_attn_heads=args.n_head,
            vocab_size=tokenizer.vocab_size,
        )
        _model = memformer

    elif args.model_arch == "xl_net":
        from notebooks.xl_net import xl_net

        config_dict = dict(
            d_model=args.dim, n_layer=args.n_layer, n_head=args.n_head, d_inner=args.dim * 4, vocab_size=tokenizer.vocab_size
        )
        _model = xl_net

    elif args.model_arch == "g_mlp_gpt":
        from notebooks.g_mlp_gpt import g_mlp_gpt

        config_dict = dict(vocab_size=tokenizer.vocab_size, seq_len=args.seqlen, dim=args.dim, depth=args.n_layer, n_head=args.n_head)
        _model = g_mlp_gpt

    elif args.model_arch == "xcit":
        from notebooks.xcit import model as xcit

        config_dict = dict(vocab_size=tokenizer.vocab_size, dim=args.dim, depth=args.n_layer, n_head=args.n_head, seq_len=args.seqlen)
        _model = xcit

    elif args.model_arch == "expire_span":
        from notebooks.expire_span import model as expire_span

        config_dict = dict(
            dim=args.dim,
            depth=args.n_layer,
            heads=args.n_head,
            seq_len=args.seqlen,
            max_mem_len=args.dim * 2,
            ramp_len=args.dim,
            vocab_size=tokenizer.vocab_size,
            dropout=0.0,
            expire_loss_coef=1e-6,
        )
        _model = expire_span

    elif args.model_arch == "nystromformer":
        from notebooks.nystromformer import nystromformer

        config_dict = dict(
            dim=args.dim,
            dim_head=args.dim,
            heads=args.n_head,
            depth=args.n_layer,
            num_landmarks=args.dim * 2,
            pinv_iterations=6,
            vocab_size=tokenizer.vocab_size,
        )
        _model = nystromformer

    elif args.model_arch == "feedback_transformer":
        from notebooks.feedback_transformer import feedback_transformer

        config_dict = dict(
            dim=args.dim,
            depth=args.n_layer,
            seq_len=args.seqlen,
            mem_len=args.dim * 4,
            dim_head=args.dim,
            n_head=args.n_head,
            attn_dropout=0.1,
            ff_dropout=0.1,
            vocab_size=tokenizer.vocab_size,
        )
        _model = feedback_transformer
    else:
        raise ValueError("Invalid Model Architecture Name")

    model = _model.get_model(**config_dict)

    return model


if __name__ == "__main__":
    args = ArgumentParser(description="Simple Binary GPT trainer")
    args.add_argument("--dim", type=int, default=32, help="Model dimension")
    args.add_argument("--n_layer", type=int, default=32, help="Number of layers in the model")
    args.add_argument("--seqlen", type=int, default=5, help="Sequence length to process")
    args.add_argument("--n_head", type=int, default=8, help="Number of Heads")

    args.add_argument("--optim", type=str, default="AdamW", choices=["AdamW", "Adafactor", "SGD"], help="Optimizer")
    args.add_argument("--lr", type=float, default=0.0001, help="Learning Rate for training")  # *
    args.add_argument("--scheduler", type=str, default="Noam", choices=["Noam", "CosineDecay"], help="LR Scheduler")
    args.add_argument("--warmup", type=float, default=0.05, help="warmup percentage")  # *
    args.add_argument("--beta1", type=float, default=0.9, help="Beta1 for Adam / Adafactor")
    args.add_argument("--beta2", type=float, default=0.999, help="Beta2 for Adam")

    args.add_argument("--save_folder", type=str, default=None, help="Folder for this model")
    args.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    args.add_argument("--gas", type=int, default=1, help="Gradient Accumulation Steps")  # *
    args.add_argument("--train_steps", type=int, default=int(50), help="Number of steps to train")
    args.add_argument("--test_every", type=int, default=int(24), help="Run evaluation after this steps")
    args.add_argument("--patience", type=int, default=5, help="Patience for training")

    args.add_argument(
        "--model_arch",
        type=str,
        default="gpt2",
        choices=["memformer", "expire_span", "xcit", "bart", "xl_net", "g_mlp_gpt", "nystromformer", "feedback_transformer", "gpt2"],
    )

    args = args.parse_args()

    # Read all the text files in the folder
    files = sorted(glob("*.txt"))
    save_folder = "./outputs"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Initialise Train and Test Data Objects
    train_dict = {"a": ["./file2.txt"], "c": ["./file3.txt"]}
    test_dict = {"b": ["./test.txt"]}

    train_data = Data(file_dict=train_dict, seqlen=args.seqlen, batch_size=args.batch_size)

    test_data = Data(file_dict=test_dict, seqlen=args.seqlen, batch_size=args.batch_size)

    model = get_model(args, tokenizer)
    # define the optimizer
    if args.optim == "SGD":
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim == "AdamW":
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optim == "Adafactor":
        from transformers.optimization import Adafactor

        optim = Adafactor(model.parameters(), scale_parameter=True, beta1=args.beta1, relative_step=True, warmup_init=True, lr=None)

    warmup = int(args.warmup * args.train_steps)

    # Initialise scheduler
    if args.scheduler == "Noam":

        def lr_lambda(current_step):
            m = min(max(1, current_step) ** -0.5, current_step * (warmup ** -1.5))
            return 100 * (128 ** -0.5) * m

        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    elif args.scheduler == "CosineDecay":

        def lr_lambda(current_step):
            if current_step < warmup:
                return float(current_step) / float(max(1, warmup))
            progress = float(current_step - warmup) / float(max(1, total_steps - warmup))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))  # * 1000

        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    def lr_lambda(current_step):
        m = min(max(1, current_step) ** -0.5, current_step * (warmup ** -1.5))
        return 100 * (128 ** -0.5) * m

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # define trainer
    tc = TrainerConfig(
        train_steps=args.train_steps,
        gas=args.gas,
        patience=args.patience,
        test_every=args.test_every,
    )
    t = Trainer(
        model=model,
        train_conf=tc,
        train_dl=train_data,
        test_dl=test_data,
        optim=optim,
        save_folder=save_folder,
        lr_scheduler=scheduler,
        wandb=False,
    )
    t.train()
