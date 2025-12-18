from datasets import load_dataset
import pandas as pd

def load_lmarena(split="train"):
    ds = load_dataset("lmarena-ai/arena-human-preference-100k")
    return ds[split].to_pandas()
