"""
Script responsável pelo balanceamento dos datasets

"""
# Importação de bibliotecas necessárias
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
    """
    Cria o analisador de argumentos para o script.
    
    Retorna:
    - argparse.Namespace: Um objeto contendo os argumentos de linha de comando.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, required=True, default=None, help='Arquivo de entrada contendo as amostras')
    parser.add_argument('-o', '--output_file', type=str, required=True, default=None, help='Nome do arquivo de saída para salvar os resultados')
    
    return parser.parse_args()

def find_least_common_class(y_labels):
    """
    Encontra a classe que possui o menor número de amostras (maligno ou benigno).
    
    Parâmetros:
    - y_labels: Lista com os rótulos das classes.
    
    Retorna:
    - tuple: Classe menos comum e sua contagem.
    """
    # Conta a frequência de cada classe em y_labels
    counter = Counter(y_labels)
    
    # Encontra a classe com a menor contagem
    least_common_class, least_common_count = counter.most_common()[-1]
    
    return (least_common_class, least_common_count)

def balance_classes(data, MAX_INSTANCES):
    """
    Equilibra (50% maligno, 50% benignos) as classes subamostrando cada classe para um número máximo de instâncias especificado.
    
    Parâmetros:
    - data: DataFrame contendo as amostras e seus rótulos.
    - MAX_INSTANCES: O número máximo de instâncias permitido por classe.
    
    Retorna:
    - DataFrame: Um DataFrame com as classes balanceadas.
    """
    # Encontra as classes únicas no DataFrame
    unique_classes = np.unique(data['class'])
    print(unique_classes)
    
    # Encontra a classe menos comum e sua contagem
    min_class, min_value = find_least_common_class(y_labels=data['class'])
    print(min_class)
    print(min_value)
    
    # Atualiza MAX_INSTANCES se a classe menos comum tiver menos instâncias do que o valor especificado
    if min_value < MAX_INSTANCES:
        print(f'Classe {min_class} com apenas {min_value} amostras. Atualizando MAX_INSTANCES para {min_value}.')
        MAX_INSTANCES = min_value
    
    # Inicializa um DataFrame vazio para armazenar as amostras subamostradas
    under_sample = pd.DataFrame()
    
    # Subamostra a primeira classe
    group1 = data[data['class'] == 0]
    sampled1 = group1.sample(MAX_INSTANCES, replace=False)
    print(sampled1)
    
    # Subamostra a segunda classe
    group2 = data[data['class'] == 1]
    sampled2 = group2.sample(MAX_INSTANCES, replace=False)
    
    # Concatena as amostras subamostradas de cada classe
    under_sample = pd.concat([sampled1, sampled2], ignore_index=True)
    
    return under_sample.reset_index(drop=True)

def process_column(column_names, delimiter='.'):
    """
    Processa os nomes das colunas, removendo caracteres indesejados e formatando-os.
    
    Parâmetros:
    - column_names: Lista com os nomes das colunas.
    - delimiter: Delimitador usado para dividir os nomes das colunas.
    
    Retorna:
    - process_col: Lista com os nomes das colunas processados.
    """
    # Processa cada nome de coluna
    process_col = [col.strip().replace(';', '.').replace(',', '.').replace(':', '.').replace('->', '').replace('/', '.').replace(' ', '_').lower() for col in column_names]

    return [col.split(delimiter)[-1] for col in process_col]

if __name__ == "__main__":
    # Cria o analisador de argumentos e obtém os argumentos de linha de comando
    arguments = create_argparse()
    
    # Lê o arquivo de entrada
    data = pd.read_csv(arguments.input_file)
    
    # Processa os nomes das colunas
    column_names = process_column(column_names=data.columns.values, delimiter='.')
    data.columns = column_names
    
    # Nomes das classes
    class_names = ['authentic', 'malware']
    
    # Nome do dataset de saída
    dataset_name = arguments.output_file
    
    # Equilibra as classes no dataset
    balanced_data = balance_classes(data=data, MAX_INSTANCES=10000)
    print(balanced_data)
    
    # Remove a primeira coluna do DataFrame equilibrado
    balanced_data.drop(columns=balanced_data.columns[0], axis=1, inplace=True)
    
    # Salva o DataFrame equilibrado em um arquivo CSV
    balanced_data.to_csv(join(f'{dataset_name}-balanced.csv'), index=False)
