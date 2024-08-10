#!/bin/bash

# Default values
batch=256
epochs=100
layers=512
folds=2

# User inputs
read -p "Input batch size: " batch
read -p "Input number of epochs: " epochs
read -p "Input the values of layers: " layers
read -p "Number of folds: " folds
if [[ -z "$batch" ]]; then
batch=256
fi
if [[ -z "$epochs" ]]; then
epochs=100
fi
if [[ -z "$layers" ]]; then
layers=512
fi
if [[ -z "$folds" ]];then
folds=2
fi
# Get dataset names
names=$(ls datasets/validation/ | grep validation.txt | cut -d "-" -f 1)

# Process each dataset
for name in $names; do
    s="datasets/validation/${name}-validation.txt"
    r="datasets/${name}-balanced.csv"
    num_samples=$(head -n 1 "$s" | cut -d " " -f 5)

    if [[ "$layers" == *","* ]]; then
        lay1=$(echo $layers | cut -d "," -f 1)
        lay2=$(echo $layers | cut -d "," -f 2)
        pipenv run python3 main.py -i "$r" -o "$name" \
            --num_samples_class_benign "$num_samples" \
            --num_samples_class_malware "$num_samples" \
            --batch_size "$batch" \
            --number_epochs "$epochs" \
            --dense_layer_sizes_g "$lay1,$lay2" \
            --dense_layer_sizes_d "$lay2,$lay1" \
            --k_fold "$folds"
    else
        pipenv run python3 main.py -i "$r" -o "$name" \
            --num_samples_class_benign "$num_samples" \
            --num_samples_class_malware "$num_samples" \
            --batch_size "$batch" \
            --number_epochs "$epochs" \
            --dense_layer_sizes_g "$layers" \
            --dense_layer_sizes_d "$layers" \
            --k_fold "$folds"
    fi

    echo "Executed for $name"
done

