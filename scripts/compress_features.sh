#!/bin/bash
DATA_DIR='/data/owcl_data/intermediate_features_npy'

# datasets=("UCF101" "DTD" "ImageNet")
datasets=("CIFAR100" "SUN397" "EuroSAT" "OxfordIIITPet" "Flowers102" "FGVCAircraft" "StanfordCars" "Food101")

length=${#datasets[@]}

# class incremental
for (( i=0; i<$length; i++ )); do
    python encode_features/compress_clip_intermediate_features.py \
        --backbone ViT-B/32 \
        --datasets ${datasets[$i]} \
        --data_root ${DATA_DIR} \
        --train_transformer_block_index 11
done
