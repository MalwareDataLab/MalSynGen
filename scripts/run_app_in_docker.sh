#!/bin/bash
cd /SynTabData/
pipenv run python3 main.py -i datasets/kronodroid_real_device-balanced.csv  --num_samples_class_benign 10000 --num_samples_class_malware 10000 --batch_size 256 --number_epochs 500 --dense_layer_sizes_g 4096 --dense_layer_sizes_d 2048 --k_fold 10
bash
