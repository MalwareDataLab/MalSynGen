"""
Script responsável pela validação dos datasets balanceados
"""
# Importação de bibliotecas necessárias
import json
import os
import sys
import datetime
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_argparse():
    """
    Cria o analisador de argumentos para o script.
    
    Retorna:
    - parser.parse_args: Um objeto contendo os argumentos de linha de comando.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, required=True, default=None, help='Arquivo de entrada contendo as amostras')
    parser.add_argument('-o', '--output_file', type=str, required=True, default=None, help='Nome do arquivo de saída para salvar os resultados')
    return parser.parse_args()

def validation(arguments):
    """
    Verifica se o arquivo de entrada é válido de acordo com as especificações é gera um arquivo de saída com base na análise
    
    Parâmetros:
    - arguments: Objeto contendo os argumentos de linha de comando.
    
    """
    # Verifica se o arquivo de entrada não é None
    if arguments.input_file is not None:
        # Lê o arquivo de entrada
        samples_file = pd.read_csv(arguments.input_file)
        
        # Abre o arquivo de saída para escrita
        f = open(arguments.output_file, "w")
        
        # Verifica se o conjunto de dados está vazio
        if samples_file.empty:
            print("empty dataset", file=f)
            return 1
        
        # Obtém os nomes das colunas do conjunto de dados
        columns = samples_file.columns.values.tolist()
        
        # Inicializa listas para armazenar colunas com valores ausentes ou incorretos
        list_of_columns_missing_values = []
        list_of_columns_with_wrong_values = []
        
        # Itera sobre cada coluna para verificar valores ausentes e incorretos
        for values in columns:
            if samples_file[values].isnull().any()==True:
                e = samples_file[samples_file[values].isnull()].index.tolist()
                e.append(values)
                list_of_columns_missing_values.append(e)
            if samples_file[values].isin([0, 1]).all()==False:
                e = samples_file[(samples_file[values] != 0) & (samples_file[values] != 1)].index.tolist()
                e.append(values)
                list_of_columns_with_wrong_values.append(e)
        
        # Imprime colunas com valores ausentes no arquivo de saída
        if list_of_columns_missing_values:
            for column in list_of_columns_missing_values:
                print("A coluna ",column[-1], "possui as seguintes entradas vazias",column[:-1],file=f )
        
        # Imprime colunas com valores incorretos no arquivo de saída
        if list_of_columns_with_wrong_values:
            for column in list_of_columns_with_wrong_values:
                print("a coluna ",column[-1], "possui as seguintes entradas não iguais aos valores esperados",file=f)	
        
        # Conta o número de amostras de malware e benignas
        malware_samples = samples_file['class'].value_counts()[1]
        benign_samples = samples_file['class'].value_counts()[0]
        
        # Imprime o número de amostras de malware e benignas no arquivo de saída
        print("Number of malware samples: ", malware_samples, file=f)
        print("Number of benign samples: ", benign_samples, file=f)
        
        # Imprime o número de colunas e linhas no arquivo de saída
        print("Número de colunas: ", len(samples_file.columns), " Número de linhas: ", len(samples_file), file=f)
        
        # Fecha o arquivo de saída
        f.close()

if __name__ == "__main__":
    # Cria o analisador de argumentos e obtém os argumentos de linha de comando
    arguments = create_argparse()
    
    # Chama a função de validação passando os argumentos
    validation(arguments)
