#!/bin/bash
OUTPUT_DIR='./results'
DATA_DIR='/data/owcl_data/intermediate_features_npy'

# Define the arrays
datasets=("CIFAR100" "SUN397" "EuroSAT" "OxfordIIITPet" "Flowers102" "FGVCAircraft" "StanfordCars" "Food101")
num_classes=(20 80 2 8 20 20 40 20)

length=${#datasets[@]}
ema_decay=0.99

# class incremental
for (( i=0; i<1; i++ )); do
    python main.py \
        --backbone ViT-B/32 \
        --incremental class \
        --datasets ${datasets[$i]} \
        --num_classes ${num_classes[$i]} \
        --held_out_dataset DTD \
        --data_root ${DATA_DIR} \
        --lr 6e-4 --weight_decay 0.05 --batch_size 2048 \
        --n_epochs 10 \
        --X_format feature \
        --node_capacity 150000000 \
        --criteria osce_other \
        --include_the_other_class \
        --other_class_calibration_loss_weight 0.1 \
        --sampler_type class_balanced \
        --wake_bs 32 \
        --wake_evaluation_iter_ratio 0.25 \
        --results_dir ${OUTPUT_DIR} \
        --learning_strategy online \
        --csv_file class_incremental.csv \
        --ema_exemplar_per_class_acc \
        --ema_exemplar_per_class_acc_decay ${ema_decay} \
        --need_compress True \
        --exp_name AnytimeCL_class_incremental
done
