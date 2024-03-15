
# enwik8, character-level

The enwik8 dataset from http://prize.hutter1.net/index.htm

The `enwik8_256` dataset is identical to `enwik8` but uses
a fixed vocacabulary of 256 bytes instead of restricting
to the 205 unique characters found in the dataset.

After running `prepare.py`:

- train.bin has 90,000,000 bytes
- val.bin has 5,000,000 bytes
- test.bin has 5,000,000 bytes
