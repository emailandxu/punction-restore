
import numpy as np


class Vocab:
    def __init__(self, vocab_path):
        self.token_list = []
        self.load_vocab(vocab_path)
        self.pad_index = 0
        self.start_index = 1
        self.end_index = 2
        self.unk_index = 3

    def __len__(self):
        return len(self.token_list)
    
    def __getitem__(self, idx):
        return self.token_list[idx]

    def load_vocab(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            tokens = [token.strip("\n") for token in f if token !=""]
            self.token_list.extend(tokens)
        
        self.char_to_index = dict(zip(self.token_list, range(len(self.token_list))))
        
    def __call__(self, char):
        return self.char_to_index.get(char, self.unk_index)
