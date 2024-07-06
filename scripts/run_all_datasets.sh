#!/bin/bash
[ -d outputs ] || { mkdir outputs; }

for DATASET in datasets/*
do
	pipenv run python3 generation.py --input_dataset  datasets/$DATASET --data_type float32 --num_samples_class_malware 10000 --num_samples_class_benign 10000 --number_epochs 1000 --classifier knn --k_fold 5 --latent_dimension 128 --output_dataset outputs/synthetic_dataset_$DATASET 
done
