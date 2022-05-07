from pathlib import Path
from dataclasses import Field, dataclass, field
from vocab.vocab import Vocab
import numpy as np
import tensorflow as tf

class PRDataset():
    def __init__(self, ds, vocab:Vocab):
        self.ds = ds
        self.vocab = vocab

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sentence, punction = self.ds[idx].strip().split("\t")
        toks = list(map(self.vocab, sentence.split(" ")))
        pucs = list(map(int, punction.split(" ")))

        toks = np.array(toks).astype(np.uint16)
        pucs = np.array(pucs).astype(np.uint8)
        return toks, pucs

    def tfds(self):
        return tf.data.Dataset.from_generator(
            lambda : self, 
            output_types=(tf.uint16, tf.uint8),
            output_shapes=(tf.TensorShape([None]), tf.TensorShape([None]))
        )

@dataclass
class Dataset():
    data_path : str
    corpus_name : str
    train_prefix : str
    dev_prefix : str
    test_prefix : str
    vocab : Vocab
    quick_debug : bool = field(default=False)

    def load_pr_lines(self, filepath):
        with open(filepath) as f:
            if self.quick_debug:
                ds = []
                for idx, line in enumerate(f):
                    if idx > 3000:
                        break
                    ds.append(line)
            else:
                ds = list(f)

            return PRDataset(ds, self.vocab)

    def __post_init__(self):
        root = Path(self.data_path)
        self.train = self.load_pr_lines(root.joinpath(f"{self.corpus_name}_{self.train_prefix}.txt").absolute())
        self.dev = self.load_pr_lines(root.joinpath(f"{self.corpus_name}_{self.dev_prefix}.txt").absolute())
        self.test = self.load_pr_lines(root.joinpath(f"{self.corpus_name}_{self.test_prefix}.txt").absolute())