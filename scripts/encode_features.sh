#!/bin/bash
SOURCE_DIR='/data/owcl_data'
CLIP_DEST_DIR='/data/owcl_data/temporal'
DINO_DEST_DIR='/data/owcl_data/dino_temporal'

# Define the arrays
datasets=("CIFAR100" "SUN397" "EuroSAT" "OxfordIIITPet" "Flowers102" "FGVCAircraft" "StanfordCars" "Food101")
length=${#datasets[@]}

# for clip
for (( i=0; i<$length; i++ )); do
    python encode_features/encode_clip_intermediate_features.py \
        --backbone ViT-B/32 \
        --datasets ${datasets[$i]} \
        --data_root ${SOURCE_DIR} \
        --store_folder ${CLIP_DEST_DIR} \
        --train_transformer_block_index 11 \
        --subsets train_test
done

# for dinov2
for (( i=0; i<$length; i++ )); do
    python encode_features/encode_dino_intermediate_features.py \
        --dinov2_config_file eval/vitb14_pretrain \
        --datasets ${datasets[$i]} \
        --data_root ${SOURCE_DIR} \
        --store_folder ${DINO_DEST_DIR} \
        --dinov2_train_transformer_block_to_last_index 1 \
        --subsets train_test
done