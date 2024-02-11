"""
Sample from a trained model
"""
import math
import os
import json
import pickle
import numpy as np
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import tiktoken
from tqdm import tqdm
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = (
    "resume"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
)
out_dir = "out"  # ignored if init_from is not 'resume'
start = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
seed = 1337
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
batch_size = 12
max_batches = -1
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster
exec(open("configurator.py").read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# model
if init_from == "resume":
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith("gpt2"):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

# poor man's data loader
dataset = checkpoint["config"]["dataset"]
data_dir = os.path.join("data", dataset)
test_data = np.memmap(os.path.join(data_dir, "test.bin"), dtype=np.uint16, mode="r")
block_size = checkpoint["config"]["block_size"]
data = test_data
last_batch = len(data) - (batch_size + -block_size + 1)
data = data[last_batch:]
num_batches = math.ceil((len(data) - block_size - 1) / batch_size)
if max_batches > 0:
    num_batches = min(num_batches, max_batches)


def get_batch():
    batch_idx = 0
    for i in tqdm(
        range(0, len(data) - (batch_size + block_size), batch_size), total=num_batches
    ):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        B = min(batch_size, len(data) - i)
        seq = data[i : i + block_size + B].astype(np.int64)
        x = np.lib.stride_tricks.as_strided(
            seq[: block_size + B - 1],
            shape=(B, block_size),
            strides=(seq.strides[0], seq.strides[0]),
        )
        y = seq[block_size : block_size + B]
        # pass to torch
        x = torch.from_numpy(x).contiguous()
        y = torch.from_numpy(y).contiguous()
        # single character prediction at test time ?
        if device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                device, non_blocking=True
            )
        else:
            x, y = x.to(device), y.to(device)
        yield x, y
        batch_idx += 1


model.eval()
model.to(device)
if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if (
    init_from == "resume"
    and "config" in checkpoint
    and "dataset" in checkpoint["config"]
):  # older checkpoints might not have these...
    meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# run generation
with torch.no_grad():
    with ctx:
        acc_loss = 0.0
        num_samples = 0
        for X, Y in get_batch():
            logits, _ = model(X)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                Y,
                reduction="sum",
            ) / math.log(2)
            acc_loss += loss.item()
            num_samples += X.shape[0]
        print(f"total loss: {acc_loss / num_samples:.4f} bpc")

with open(os.path.join(out_dir, "test.json"), "w") as f:
    json.dump(
        {
            "bpc": acc_loss / num_samples,
            "num_samples": num_samples,
        },
        indent=2,
    )
