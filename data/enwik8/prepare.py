"""
Prepare the enwik8 dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map bytes to ints.
"""
import io
import os
import pickle
import zipfile

import numpy as np
import requests

# download the enwik8 dataset
input_file_path = os.path.join(os.path.dirname(__file__), "enwik8")
if not os.path.exists(input_file_path):
    data_url = "http://mattmahoney.net/dc/enwik8.zip"
    buf = io.BytesIO(requests.get(data_url).content)
    with zipfile.ZipFile(buf, "r") as zip_ref:
        zip_ref.extractall(os.path.dirname(__file__))

with open(input_file_path, "rb") as f:
    data = f.read()
print(f"length of dataset in bytes: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
# print("all the unique characters:", "".join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    if isinstance(s, str):
        s = s.encode("utf-8")
    return [stoi[c] for c in s]  # encoder: take a string, output a list of integers


def decode(l):
    b_array = b"".join([itos[i].to_bytes(1, "big") for i in l])
    return b_array.decode("utf-8")  # decoder: take a list of integers, output a string


print(decode(encode(data[:100])))

# create the train and test splits
n = len(data)
n_train = int(n * 0.9)
n_val = int(n * 0.05)
n_test = n - n_train - n_val
train_data = data[:n_train]
val_data = data[n_train : n_train + n_val]
test_data = data[n_train + n_val :]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
test_ids = encode(test_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
print(f"test has {len(test_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
test_ids = np.array(test_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))
test_ids.tofile(os.path.join(os.path.dirname(__file__), "test.bin"))

# save the meta information as well, to help us encode/decode later
meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
}
with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

# length of dataset in bytes: 100,000,000
# vocab size: 205
# <mediawiki xmlns="http://www.mediawiki.org/xml/export-0.3/" xmlns:xsi="http://www.w3.org/2001/XMLSch
# train has 90,000,000 tokens
# val has 5,000,000 tokens
# test has 5,000,000 tokens
