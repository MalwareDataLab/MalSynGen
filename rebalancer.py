from collections import Counter
import json
import os
import sys
import datetime
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join

def create_argparse(): 
          parser=argparse.ArgumentParser()
          parser.add_argument('-i','--input_file',type=str,required=True,default=None ,help='Input file contaning thew samples')
          parser.add_argument("-o","--output_file",type=str,required=True,default=None, help="Name of the output file for saving the results")
          
          return  parser.parse_args()

def find_least_common_class(y_labels):
    # Count the frequency of each class in y_labels
    counter = Counter(y_labels)
    
    # Find the class with the minimum count
    least_common_class, least_common_count = counter.most_common()[-1]
    
    return (least_common_class, least_common_count)
    
def balance_classes(data, MAX_INSTANCES):
    """
    Balances classes by under-sampling each class to a specified maximum number of instances.
    
    Parameters:
    - X: Features, numpy array of shape (n_samples, n_features)
    - y_ohe: One-hot encoded labels, numpy array of shape (n_samples, n_classes)
    - label_list: List of actual labels corresponding to each sample
    - MAX_INSTANCES: The maximum number of instances allowed per class.
    
    Returns:
    - A dictionary with balanced features, labels, one-hot encoded labels, actual label list, and indices.
    """
    # y = data_dict['labels']['labels']
    unique_classes = np.unique(data['class'])
    print(unique_classes)
    min_class, min_value = find_least_common_class(y_labels=data['class'])
    print(min_class)
    print(min_value)
    if min_value < MAX_INSTANCES:
        print(f'Class {min_class} with only {min_value} samples. Updating MAX_INSTANCES to {min_value}.')
        MAX_INSTANCES = min_value
    
    under_sample = pd.DataFrame()
    # np.random.seed(0)  # For reproducibility
    # Find indices of the current class
    group1 = data[data['class']==0]
    sampled1 = group1.sample(MAX_INSTANCES, replace=False)
    print(sampled1)
    group2 = data[data['class']==1]
    sampled2 = group2.sample(MAX_INSTANCES, replace=False)
    # Concatenate all sampled indices from each class
    
    under_sample = pd.concat([sampled1, sampled2],ignore_index=True)
    
    return under_sample.reset_index(drop=True)

def process_column(column_names, delimiter='.'):
    process_col = [col.strip().replace(';','.').replace(',','.').replace(':','.').replace('->','').replace('/','.').replace(' ', '_').lower() for col in column_names]

    return [col.split(delimiter)[-1] for col in process_col]
if __name__ == "__main__":
        arguments=create_argparse()
        data = pd.read_csv(arguments.input_file)
        column_names = process_column(column_names=data.columns.values, delimiter='.')
        data.columns = column_names
        class_names=['authentic', 'malware']
        
        dataset_name = arguments.output_file
        balanced_data = balance_classes(data=data, MAX_INSTANCES=10000)
        print(balanced_data)
        balanced_data.drop(columns=balanced_data.columns[0], axis=1, inplace=True)
        balanced_data.to_csv(join( f'{dataset_name}-balanced.csv'),index=False)
