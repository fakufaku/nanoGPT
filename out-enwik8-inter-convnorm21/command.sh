python ./train.py config/train_enwik8.py \
  --out_dir=out-enwik8-inter-convnorm21 \
  --num_future_targets=0 \
  --n_embd=768 \
  --n_layer=8 \
  --n_head=8 \
  --inter_weights="1:0.1,3:0.1,5:0.1,7:0.1" \
  --selfcond=False \
  --lr_decay_iters=20000 --max_iters=20000 \
  --use_conv_norm=True --conv_norm_kernel=21
