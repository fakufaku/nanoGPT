# NanoGPT on Enwik8

## Environment Preparation

```
conda create -n nanogpt python=3.10
conda activate nanogpt
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

## Prepare the dataset

The original data is stored [here](http://mattmahoney.net/dc/enwik8.zip).

Run the following command
```
cd data/enwik8
python data/enwik8/prepare.py
```
which should give as output a list of `6064` tokens
and the number of tokens per split:
```
length of dataset in characters: 99,621,832
all the unique characters:
<long list of characters from several alphabests>
vocab size: 6,064
train has 89,659,648 tokens
val has 4,981,091 tokens
test has 4,981,093 tokens
```
The dataset is stored in `data/enwik8` in files `train.bin`, `val.bin`, and `test.bin`.
