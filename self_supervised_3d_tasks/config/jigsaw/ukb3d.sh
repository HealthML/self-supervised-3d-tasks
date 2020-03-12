#!/bin/bash -eu
#!/bin/sh

python train_and_eval.py \
  --workdir ~/workspace/self-supervised-transfer-learning/ \
  --dataset_dir /mnt/mpws2019cl1/brain_mri/tf_records/ \
  --task jigsaw \
  --dataset ukb3d \
  --train_split train \
  --val_split val \
  --batch_size 2 \
  --eval_batch_size 2 \
  \
  --architecture unet_resnet3d_class \
  --filters_factor 2 \
  --optimizer adam \
  \
  --preprocessing duplicate_channels3d,crop_patches3d \
  --patches_per_side 3 \
  --patch_jitter 10 \
  --perm_subset_size 8 \
  --embed_dim 1000 \
  \
  --lr 0.0005 \
  --lr_scale_batch_size 8 \
  --epochs 100 \
  --warmup_epochs 5 \
  \
  --serving_input_shape None,32,32,32,4 \
  --save_checkpoints_secs 200 \
  --throttle_secs 400 \
  --keep_checkpoint_every_n_hours 48 \
  "$@"

