import string
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from underthesea import word_tokenize



class Tokenizer:
    def __init__(self, src_path, tgt_path):
        self.src_tokenizer = get_tokenizer("basic_english")
        self.tgt_tokenizer = lambda text: word_tokenize(text, format="text").split()
        
        self.src_vocab = self.build_vocab(src_path, self.src_tokenizer)
        self.tgt_vocab = self.build_vocab(tgt_path, self.tgt_tokenizer)
    
    def build_vocab(self, file_path, tokenizer):
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.readlines()
            #preprocess
            data = [line.lower() for line in data]
            data = [line.strip() for line in data]
            # remove puchuation
            data = [line.translate(str.maketrans('', '', string.punctuation)) for line in data]
            
        
        def yeild_tokens(data):
            for line in data:
                yield tokenizer(line)
                
        
    
        vocab = build_vocab_from_iterator(yeild_tokens(data), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
        vocab.set_default_index(vocab["<unk>"])
        return vocab
    
    def tokenize(self, src_text, tgt_text):
        src_tokens = self.src_tokenizer(src_text)
        tgt_tokens = self.tgt_tokenizer(tgt_text)
        
        src_tokens = [self.src_vocab["<bos>"]] + self.src_vocab(src_tokens) + [self.src_vocab["<eos>"]]
        tgt_tokens = [self.tgt_vocab["<bos>"]] + self.tgt_vocab(tgt_tokens) + [self.tgt_vocab["<eos>"]]
        return src_tokens, tgt_tokens
    
    
    
# tokenizer = Tokenizer("data/en_sents", "data/vi_sents")

# src_text = "I am a student"
# tgt_text = "Tôi là sinh viên"
# src_tokens, tgt_tokens = tokenizer.tokenize(src_text, tgt_text)
# print(src_tokens, tgt_tokens)