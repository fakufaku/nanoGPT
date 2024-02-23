python ./train.py config/train_enwik8.py \
  --out_dir=out-enwik8 \
  --n_embd=768 \
  --n_layer=8 \
  --n_head=8 \
  --inter_weights="" \
  --selfcond=False \
  --lr_decay_iters=20000 \
  --max_iters=20000
