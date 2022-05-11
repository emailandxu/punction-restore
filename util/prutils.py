from pathlib import Path
from configs.config import Config
from functools import lru_cache

def _to_sign(t):
  return ["", " ", ", ", ". ", "! ","? "][t]

def add_sign_by_mask(chars, masks):
  return "".join([chars[idx] + _to_sign(t) for idx, t in enumerate(masks)])

@lru_cache(maxsize=1)
def get_test_samples(config:Config, n=10):
  root = Path(config.dataset_config["data_path"])
  filename = f"{config.dataset_config['corpus_name']}_{config.dataset_config['test_prefix']}.txt"
  with open(root.joinpath(filename).absolute(), "r") as f:
    for idx, line in enumerate(map(lambda line:line.strip(),f)):
      if idx > n:
        break
      yield line