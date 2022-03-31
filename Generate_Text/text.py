from base64 import decode
import nbox
from nbox import Operator
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def hr():
    print("\n", "=" * 80, "\n")

class GenerateText(Operator):
    def __init__(self, parent, file_names):
        super().__init__()
        self.parent = parent
        self.file_names = file_names
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2') 

    def forward(self, num_gen=500):
        ctx_len = 500
        for name in self.file_names:
            hr()
            print("Processing: ", name)
            hr()
            f = open("{}/{}".format(self.parent, name), "r+")
            text = f.read()
            start = len(text)
            if start>ctx_len:
                text = text[-ctx_len:]
                start = ctx_len
            encoded_input = self.tokenizer.encode(text, return_tensors='pt')
            outputs = self.model.generate(
                encoded_input,
                num_beams=5,
                max_length=num_gen,
                no_repeat_ngram_size=2,
                early_stopping=False
                )
            decoded = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False,
            )
  
            f.write(decoded[start: ])
            f.close()
            hr()
            print("Generation complete for: ", name)

class Text(Operator):
    def __init__(self, parent, file_names):
        super().__init__()
        self.generator = GenerateText(parent, file_names)
    
    def forward(self, num_gen=500):
        self.generator(num_gen=num_gen)