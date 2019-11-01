#!/bin/bash -eu
#!/bin/sh

python train_and_eval.py \
  --workdir ~/workspace/self-supervised-transfer-learning/ \
  --checkpoint_dir ~/workspace/self-supervised-transfer-learning/checkpoints/ \
  --dataset_dir /mnt/30T/brats/ \
  --task supervised_segmentation \
  --dataset brats_supervised_3d \
  --train_split train \
  --val_split val \
  --batch_size 64 \
  --eval_batch_size 32 \
  \
  --architecture unet_resnet3d \
  --filters_factor 4 \
  --optimizer adam \
  \
  --preprocessing plain_preprocess \
  \
  --lr 0.1 \
  --lr_scale_batch_size 64 \
  --epochs 100 \
  --warmup_epochs 5 \
  \
  --serving_input_shape None,128,128,128,2 \
  --save_checkpoints_secs 60 \
  --throttle_secs 90 \
  --keep_checkpoint_every_n_hours 48 \
  "$@"
