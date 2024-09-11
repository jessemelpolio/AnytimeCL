#!/bin/bash
OUTPUT_DIR='./results'
ema_decay=0.99
DATA_DIR='/data/owcl_data/intermediate_features_npy'
DINO_DATA_DIR='/data/owcl_data/dino_intermediate_features_npy'


# DINOv2 task incremental
python main.py \
      --backbone ViT-B/32 \
      --incremental dataset \
      --data_root ${DATA_DIR} \
      --datasets CIFAR100,SUN397,EuroSAT,OxfordIIITPet,Flowers102,FGVCAircraft,StanfordCars,Food101 \
      --held_out_dataset ImageNet,UCF101,DTD \
      --lr 6e-4 --weight_decay 0.05 --batch_size 2048 \
      --n_epochs 10 \
      --X_format feature \
      --node_capacity 150000000 \
      --criteria osce_other \
      --include_the_other_class \
      --other_class_calibration_loss_weight 0.1 \
      --sampler_type class_balanced \
      --wake_bs 32 \
      --wake_evaluation_iter_ratio 10 \
      --results_dir ${OUTPUT_DIR} \
      --learning_strategy online \
      --csv_file task_incremental.csv \
      --ema_exemplar_per_class_acc \
      --ema_exemplar_per_class_acc_decay ${ema_decay} \
      --use_dino \
      --dinov2_data_root ${DINO_DATA_DIR} \
      --exp_name AnytimeCL_task_incremental_dinov2

