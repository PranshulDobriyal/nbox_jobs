import os
import torch
import unittest
import numpy as np
from glob import glob
from functools import lru_cache

# ---------
import nbox
from nbox.utils import folder, join, hash_
# ---------

# import all the required model methods
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


@lru_cache()
def get_ds():
    from gpts.common import get_text, get_tensors

    return get_tensors(get_text())


@lru_cache()
def get_tokenizer(key):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(key)


# some constants/functions that are used through out the script
DIM = 16

def get_hf_model(name, vocab_size):
    get_model = {
    "bart": get_bart_model,
    "gpt2": get_gpt2_model,
    "xlnet": get_xlnet_model,
    }
    hf_config = { 
        "bart": dict(
            vocab_size=vocab_size,
            d_model=DIM,
            encoder_layers=1,
            encoder_attention_heads=1,
            encoder_ffn_dim=DIM*2,
            decoder_ffn_dim=DIM*2,
            decoder_layers=1,
            decoder_attention_heads=1,
            max_position_embeddings=DIM*8,
            ),
        "gpt2": dict(
            vocab_size=vocab_size,
            d_model=DIM,
            n_layer=1,
            n_head=1,
            d_inner=DIM*2,
            ),
        "xlnet": dict(
            vocab_size=vocab_size,
            d_model=DIM,
            n_layer=1,
            n_head=1,
            d_inner=DIM*2,
        ),
    }
    return get_model[name](**hf_config[name])[0]

def get_lucid_model(name, seqlen, vocab_size):
    get_model = {
        "xspan": get_xspan_model,
        "fnets": get_fnets_model,
        "gmlp": get_gmlp_model,
        "linformer": get_linformer_model,
        "ftransformer": get_ftransformer_model,
        "memformer": get_memformer_model,
        "nystromformer": get_nystromformer_model,
        "xcit": get_xcit_model,
    }
    lucid_config = {
        "xspan": dict(
            dim=DIM, 
            depth=1, 
            heads=1, 
            seq_len=seqlen,
            vocab_size=vocab_size
        ),
        "ftransformer": dict(
            num_tokens=vocab_size,
            dim=DIM,
            depth=1,
            seq_len=seqlen,
            mem_len=DIM*8,
            dim_head=DIM*2,
            n_head=1,
            ),
        "fnets": dict(
            dim=DIM, 
            depth=1, 
            seq_len=seqlen, 
            vocab_size=vocab_size
        ),
        "gmlp": dict(
            dim=DIM, 
            depth=1, 
            vocab_size=vocab_size, 
            seq_len=seqlen, 
            n_head=1
        ),
        "linformer": dict(
            vocab_size=vocab_size,
            dim=DIM,
            seq_len=seqlen,
            depth=1,
            heads=1,
        ),
        "memformer": dict(
            dim=DIM, 
            num_memory_slots=seqlen//2,
            max_seq_len=seqlen, 
            vocab_size=vocab_size, 
            heads=1, 
            depth=1,
            mem_update_attn_heads=1
        ),
        "nystromformer": dict(
            dim=DIM,
            depth=1,
            seq_len=seqlen,
            dim_head=DIM*2,
            heads=1,
        ),
        "xcit": dict(
            dim=DIM, 
            depth=1, 
            n_head=1, 
            seq_len=seqlen, 
            vocab_size=vocab_size,
        )
    }
    return get_model[name](**lucid_config[name])

class TestModels(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestModels, self).__init__(*args, **kwargs)
        self.tensors, vocab = get_ds()        
        self.vocab_size = len(vocab)
        set_seed(4)

    def hf_model_test(self, m):
        vocab_size = self.vocab_size
        device = torch.device("cpu")
        out = []
        with torch.no_grad():
            model= get_hf_model(m, vocab_size)
            model.to(device)
            out = model(self.tensors)
        torch.cuda.empty_cache()
        self.assertEqual(tuple(out.logits.shape), (289, 128, 39), f"Test failed for {model}")

    def lucid_model_test(self, m):
        seqlen = 128
        vocab_size = self.vocab_size
        device = torch.device("cpu")
        model, _ = get_lucid_model(m, seqlen, vocab_size)
        model.to(device)
        out = model(self.tensors)
        if type(out) is not torch.Tensor:
            out = out[0]
        self.assertEqual(tuple(out.shape), (289, 128, 39), f"Test failed for {model}")

        

    # ========== Huggingface Model Testing ========== #

    def test_hf_bart(self):       
        self.hf_model_test("bart")
    
    def test_hf_gpt(self):
        self.hf_model_test("gpt2")

    def test_hf_xlnet(self):
        self.hf_model_test("xlnet")

    # ========== LucidRains Model Testing ========== #

    def test_xspan(self):
        self.lucid_model_test("xspan")
    
    def test_cit(self):
        self.lucid_model_test("xcit")

    def test_nystromformer(self):
        self.lucid_model_test("nystromformer")

    def test_memformer(self):
        self.lucid_model_test("memformer")
    
    def test_linformer(self):
        self.lucid_model_test("linformer")
    
    def test_gmlp(self):
        self.lucid_model_test("gmlp")
    
    def test_fnets(self):
        self.lucid_model_test("fnets")

    def test_ftransformer(self):
        self.lucid_model_test("ftransformer")


@unittest.skipUnless(torch.cuda.is_available(), "cuda unavailable")
class TestModelsCUDA(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestModelsCUDA, self).__init__(*args, **kwargs)
        self.tensors, vocab = get_ds() 
        self.tensors = self.tensors.to(torch.device("cuda"))       
        self.vocab_size = len(vocab)
        set_seed(4)

    def hf_model_test(self, m):
        vocab_size = self.vocab_size
        device = torch.device("cuda")
        out = []
        with torch.no_grad():
            model= get_hf_model(m, vocab_size)
            model.to(device)
            out = model(self.tensors)
        torch.cuda.empty_cache()
        self.assertEqual(tuple(out.logits.shape), (289, 128, 39), f"Test failed for {model}")

    def lucid_model_test(self, m):
        seqlen = 128
        vocab_size = self.vocab_size
        device = torch.device("cuda")
        model, _ = get_lucid_model(m, seqlen, vocab_size)
        model.to(device)
        out = model(self.tensors)
        if type(out) is not torch.Tensor:
            out = out[0]
        self.assertEqual(tuple(out.shape), (289, 128, 39), f"Test failed for {model}")


    # ========== Huggingface Model Testing ========== #

    def test_hf_bart_cuda(self):       
        self.hf_model_test("bart")
    
    def test_hf_gpt_cuda(self):
        self.hf_model_test("gpt2")

    def test_hf_xlnet_cuda(self):
        self.hf_model_test("xlnet")

     # ========== LucidRains Model Testing ========== #

    def test_xspan_cuda(self):
        self.lucid_model_test("xspan")
    
    def test_cit_cuda(self):
        self.lucid_model_test("xcit")

    @unittest.skip("Breaks every CUDA test performed after it")
    def test_nystromformer_cuda(self):
        self.lucid_model_test("nystromformer")

    def test_memformer_cuda(self):
        self.lucid_model_test("memformer")
    
    def test_linformer_cuda(self):
        self.lucid_model_test("linformer")
    
    def test_gmlp_cuda(self):
        self.lucid_model_test("gmlp")
    
    def test_fnets_cuda(self):
        self.lucid_model_test("fnets")

    def test_ftransformer_cuda(self):
        self.lucid_model_test("ftransformer")


class TestTokenizerRuntimeEngine(unittest.TestCase):
    def test_list(self):
        files = sorted(glob(join(folder(__file__), "gpts", "*.py")))
        tokenizer = get_tokenizer("gpt2")
        data = TokenizerRuntimeEngine(files, tokenizer, seqlen = 32, batch_size = 1)

        out = data[None]
        self.assertEqual(out["input_ids"].shape, (1, 32))
        self.assertEqual(out["labels"].shape, (1, 32))

    def test_list_int(self):
        files = sorted(glob(join(folder(__file__), "gpts", "*.py")))
        tokenizer = get_tokenizer("gpt2")
        data = TokenizerRuntimeEngine(files, tokenizer, seqlen = 32, batch_size = 1)

        out = data[4]
        self.assertEqual(out["input_ids"].shape, (4, 32))
        self.assertEqual(out["labels"].shape, (4, 32))

    def test_dict(self):
        files = sorted(glob(join(folder(__file__), "gpts", "*.py")))
        n_c = 4
        fps = {}
        for f in files:
            _c = (np.linspace(0, 1, n_c) <= np.random.random()).argmin()
            fps.setdefault(str(_c), []).append(f)

        tokenizer = get_tokenizer("gpt2")
        data = TokenizerRuntimeEngine(fps, tokenizer, seqlen = 32, batch_size = 1)

        out = data[None]
        self.assertEqual(out["input_ids"].shape, (1, 32))
        self.assertEqual(out["labels"].shape, (1, 32))

    def test_dict_int(self):
        files = sorted(glob(join(folder(__file__), "gpts", "*.py")))
        fps = {"1": files[:1], "2": files[1:4], "3": files[4:],}
        tokenizer = get_tokenizer("gpt2")
        data = TokenizerRuntimeEngine(fps, tokenizer, seqlen = 32, batch_size = 1)

        out = data[4]
        self.assertEqual(out["input_ids"].shape, (4, 32))
        self.assertEqual(out["labels"].shape, (4, 32))

    def test_dict_dict(self):
        files = sorted(glob(join(folder(__file__), "gpts", "*.py")))
        fps = {"1": files[:1], "2": files[1:4], "3": files[4:],}
        tokenizer = get_tokenizer("gpt2")
        data = TokenizerRuntimeEngine(fps, tokenizer, seqlen = 32, batch_size = 1)

        out = data[{"1": 1, "2": 2, "3": 3}]
        self.assertEqual(out["input_ids"].shape, (6, 32))
        self.assertEqual(out["labels"].shape, (6, 32))

    @unittest.expectedFailure
    def test_dict_dict_fail(self):
        files = sorted(glob(join(folder(__file__), "gpts", "*.py")))
        fps = {"1": files[:1], "2": files[1:4], "3": files[4:],}
        tokenizer = get_tokenizer("gpt2")
        data = TokenizerRuntimeEngine(fps, tokenizer, seqlen = 32, batch_size = 1)

        data[{"4"}]
