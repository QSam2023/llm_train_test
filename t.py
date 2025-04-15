import torch
from datasets import load_dataset

max_seq_length = 2048 # Supports RoPE Scaling interally, so choose any!
# Get LAION dataset
url = "./data/demo_data.json"
#url = "./data/unified_chip2.jsonl"
dataset = load_dataset("json", data_files = url, split = "train")
print(dataset[0])
for convo in dataset:
    print(convo)
    break
