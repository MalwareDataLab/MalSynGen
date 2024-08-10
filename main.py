
"""
Módulo principal para a execução de experimentos da ferramenta, responsável pelo fluxo de execução, parâmetros de entrada, log de dados e organização dos dados de saída.
Funções: 
- run_experiment : Função responsável pelo principal fluxo de execução da cGAN.
- get_adversarial_model : Função utilizada para a chamada do método de instanciamento da rede adversarial cGAN.
- show_and_export_results : Função para demonstrar e exportar resultados de métricas de fidelidade e utilidade, além da chamada de criação dos plots destas métricas.
- comparative_data : Função para a comparação entres os dados sintéticos com dados reais usando métricas de fidelidade (Similaridade de cosseno, erro quadrático médio.
- evaluate_TR_As_data, evaluate_TS_Ar_data :Cálculo das métricas de utilidade dos classificadores.
- p_value_test : A funlção calcula o valor-p (p-value) utilizando o teste de Wilcoxon para amostras pareadas das métricas de utilidade dos classificadores.
- generate_sample : Função utilizada para a geração de amostras sintéticas usando o modelo gerador da cGAN. Utilizado para gerar os datatsets S e s.
- plot_to_image : Converte um gráfico Matplotlib em uma imagem utilizável pelo TensorBoard.
- show_all_settings : Função exibe todas as configurações do experimento nos logs.
- show_model :  A função imprime os parametros da rede adversarial no log de saida
- initial_step :  A função realiza a configuração inicial do experimento; salvamento dos argumentos da linha de comando e o Carregamento do dataset e estabelecimento da forma dos dados de entrada.
- create_argparse :   Estabelece a lista de parâmetros de entradas para o argparse. 
"""

## Importação de bibliotecas necessárias
try:

    import json
    import os
    import sys
    import logging
    import datetime
    import time
    import argparse
    import warnings
    import ssl
    import csv
    import math
    from sklearn.metrics import log_loss
    from logging.handlers import RotatingFileHandler
    import mlflow
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from aim import Image, Distribution,Run
    from pathlib import Path
    import tensorflow as tf
    import scipy.stats as stats
    from tensorflow.keras.losses import BinaryCrossentropy
    from sklearn.model_selection import StratifiedKFold
    import sklearn
    from sklearn import metrics
    from sklearn.metrics import confusion_matrix
    import requests
    from Models.ConditionalGANModel import ConditionalGAN
    from Models.AdversarialModel import AdversarialModel
    from Models.Classifiers import Classifiers
    import scipy.stats
    from sklearn import svm
    import keras
    import mlflow.models
    from sklearn.linear_model import SGDRegressor
    from Tools.tools import PlotConfusionMatrix
    from Tools.tools import PlotFidelityeMetrics
    from Tools.tools import PlotCurveLoss
    from Tools.tools import ProbabilisticMetrics
    from Tools.tools import PlotClassificationMetrics
  
    from Tools.tools import DEFAULT_COLOR_NAME 
    from sklearn.linear_model import SGDClassifier
    import neptune
    from mlflow.models import infer_signature
    from tensorflow.keras.callbacks import TensorBoard
    import io
    import pickle   



    ## Definição de flags de uso de ferramentas
    USE_AIM=False
    USE_MLFLOW=False
    USE_NEPTUNE=False
    USE_TENSORBOARD=False

    os.environ["KERAS_BACKEND"] = "tensorflow"
except ImportError as error:
     ## Tratamento de erro de dependências
    print(error)
    print()
    print("1. (optional) Setup a virtual environment: ")
    print("  python3 - m venv ~/Python3env/MalSynGen")
    print("  source ~/Python3env/MalSynGen/bin/activate ")
    print()
    print("2. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)


## Configurações de log e supressão de avisos
##Os niveis de verbosidade do log podem ser INFO (1) ou DEBUG (2).
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*the default value of `keepdims` will become False.*")
    warnings.filterwarnings("ignore", message="Variables are collinear")

## Definição de configurações padrão da rede cGAN
DEFAULT_VERBOSITY = logging.INFO
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'
DEFAULT_DATA_TYPE = "float32"
##Número padrão de amostras malware a serem geradas.
DEFAULT_NUMBER_GENERATE_MALWARE_SAMPLES = 2000
## Número padrão de amostras beningas a serem geradas.
DEFAULT_NUMBER_GENERATE_BENIGN_SAMPLES = 2000
##Número padrão de épocas (iterações de treinamento) da cGAN.
DEFAULT_NUMBER_EPOCHS_CONDITIONAL_GAN = 100
##Número padrão de dobras a serem utilizados
DEFAULT_NUMBER_STRATIFICATION_FOLD = 5
##Número padrão de camadas latentes
DEFAULT_ADVERSARIAL_LATENT_DIMENSION = 128

## Algoritmo padrão de treinamento para cGAN. Opções: 'Adam', 'RMSprop', 'Adadelta'.
DEFAULT_ADVERSARIAL_TRAINING_ALGORITHM = "Adam"
##Função de ativação padrão da cGAN. Opções: 'LeakyReLU', 'ReLU', 'PRe
DEFAULT_ADVERSARIAL_ACTIVATION = "LeakyReLU"  ## ['LeakyReLU', 'ReLU', 'PReLU']
## Valor padrão para a taxa de decaimento do dropout do gerador da cGAN.
DEFAULT_ADVERSARIAL_DROPOUT_DECAY_RATE_G = 0.2
## Valor padrão para a taxa de decaimento do dropout do discriminador da cGAN.
DEFAULT_ADVERSARIAL_DROPOUT_DECAY_RATE_D = 0.4
##Valor padrão para a central da distribuição gaussiana do inicializador.
DEFAULT_ADVERSARIAL_INITIALIZER_MEAN = 0.0
## Valor padrão para desvio padrão da distribuição gaussiana do inicializador.
DEFAULT_ADVERSARIAL_INITIALIZER_DEVIATION = 0.02
##Tamanho de lota padrão da cGAN. Opções: 16, 32, 64,128,256
DEFAULT_ADVERSARIAL_BATCH_SIZE = 32
##Valor padrão para número de neurônios das camadas densas do gerador.
DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_G = [512]
##Valor padrão para número de neurônios das camadas densas do discriminador.
DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_D = [512]
##Valor padrão para a média da distribuição do ruído aleatório de entrada.
DEFAULT_ADVERSARIAL_RANDOM_LATENT_MEAN_DISTRIBUTION = 0.0
##Valor para o desvio padrão do ruído aleatório de entrada
DEFAULT_ADVERSARIAL_RANDOM_LATENT_STANDER_DEVIATION = 1.0

##Configurações a função de ativação da última camada
DEFAULT_CONDITIONAL_LAST_ACTIVATION_LAYER = "sigmoid"
##Configurações do perceptron
DEFAULT_PERCEPTRON_TRAINING_ALGORITHM = "Adam"
DEFAULT_PERCEPTRON_LOSS = "binary_crossentropy"
DEFAULT_PERCEPTRON_DENSE_LAYERS_SETTINGS = [512, 256, 256]
DEFAULT_PERCEPTRON_DROPOUT_DECAY_RATE = 0.2
DEFAULT_PERCEPTRON_METRIC = ["accuracy"]
## Valor padrão para a opção de salvar modelos: True ou False
DEFAULT_SAVE_MODELS = True
##Caminhos para os arquivos de saida
DEFAULT_OUTPUT_PATH_CONFUSION_MATRIX = "confusion_matrix"
DEFAULT_OUTPUT_PATH_TRAINING_CURVE = "training_curve"
##Classificadores utilizados por padrão. Opções: RandomForest, SupportVectorMachine, DecisionTree, AdaBoost, Perceptron, SGDRegressor, XGboost
DEFAULT_CLASSIFIER_LIST = ["RandomForest", "SupportVectorMachine","DecisionTree", "AdaBoost","Perceptron","SGDRegressor","XGboost"] 

"""
Nível de verbosidade das mensagens durante o treinanmento da cGAN
"""
DEFAULT_VERBOSE_LIST = {logging.INFO: 2, logging.DEBUG: 1, logging.WARNING: 2,
                        logging.FATAL: 0, logging.ERROR: 0}
"""
Novo do arquivo onde serão salvos os logs de execução
"""
LOGGING_FILE_NAME = "logging.log"

aim_run = None
callbacks = None
file_writer = None

## Definição de tipos de argumentos personalizados para argparse
def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def list_of_strs(arg):
    return list(map(str, arg.split(',')))

def generate_samples(instance_model, number_instances, latent_dimension, label_class, verbose_level,
                     latent_mean_distribution, latent_stander_deviation):
    """
    Função utilizada para a geração de amostras sintéticas usando o modelo gerador da cGAN. Utilizado para gerar os datatsets S e s

    Parâmetros:
    - instance_model: modelo gerador
    - number_instances: número de amostras a serem geradas
    - latent_dimension: dimensão do vetor latente
    - label_class: classe das amostras (1 para malware, 0 para benigno)
    - verbose_level: nível de verbosidade
    - latent_mean_distribution: média da distribuição latente
    - latent_stander_deviation: desvio padrão da distribuição latente

    Retorna:
    - generated_samples: amostras geradas
    - label_samples_generated: rótulos das amostras geradas
    """

    if np.ceil(label_class) == 1:
        label_samples_generated = np.ones(number_instances, dtype=np.float32)
        label_samples_generated = label_samples_generated.reshape((number_instances, 1))
    else:
        label_samples_generated = np.zeros(number_instances, dtype=np.float32)
        label_samples_generated = label_samples_generated.reshape((number_instances, 1))

    latent_noise = np.random.normal(latent_mean_distribution, latent_stander_deviation,
                                    (number_instances, latent_dimension))
    generated_samples = instance_model.generator.predict([latent_noise, label_samples_generated], verbose=verbose_level)
    generated_samples = np.rint(generated_samples)

    return generated_samples, label_samples_generated
    

def plot_to_image(figure):
    """
    Converte um gráfico Matplotlib em uma imagem utilizável pelo TensorBoard.

    Parâmetros:
    - figure: figura Matplotlib

    Retorna:
    - digit: imagem decodificada
    """
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    digit = tf.image.decode_png(buf.getvalue(), channels=4)
    digit = tf.expand_dims(digit, 0)

    return digit

def comparative_data(fold, x_synthetic, real_data, label):
    """
    Função para a comparação entres os dados sintéticos com dados reais usando métricas de fidelidade (Similaridade de cosseno, erro quadrático médio

    Parâmetros:
    - fold: índice da iteração atual
    - x_synthetic: dados sintéticos
    - real_data: dados reais
    - label: rótulo dos dados

    Retorna:
    - lista com métricas de similaridade
    """
    instance_metrics = ProbabilisticMetrics()
    synthetic_mean_squared_error = instance_metrics.get_mean_squared_error(real_data, x_synthetic)
    synthetic_cosine_similarity = instance_metrics.get_cosine_similarity(real_data, x_synthetic)
    synthetic_maximum_mean_discrepancy = instance_metrics.get_maximum_mean_discrepancy(real_data, x_synthetic)
    logging.info(f"Similarity Metrics")
    logging.info(f"  Synthetic Fold {fold + 1} {label} - Mean Squared Error: " + str(synthetic_mean_squared_error))
    logging.info(f"  Synthetic Fold {fold + 1} {label} - Cosine Similarity: " + str(synthetic_cosine_similarity))
    logging.info(f"  Synthetic Fold {fold + 1} {label} - Maximum Mean Discrepancy: " + str(synthetic_maximum_mean_discrepancy))
    logging.info("")

    return [synthetic_mean_squared_error,synthetic_cosine_similarity,synthetic_maximum_mean_discrepancy]

def evaluate_TR_As_data(list_classifiers, x_TR_As, y_TR_As, fold, k, generate_confusion_matrix,
                       output_dir, classifier_type, out_label, path_confusion_matrix, verbose_level, dict, TR_As_aucs):
    """
    Avalia os do classificador TR_As (Treinado com dados reais/ Avaliado com dados sintéticos) 

    Parâmetros:
    - list_classifiers: lista de classificadores
    - x_TR_As: dados utilizada para evaluar os classificadores TR_As
    - y_TR_As: rótulos dos dados de avaliação
    - fold: índice da iteração atual
    - k: índice da subdivisão atual
    - generate_confusion_matrix: flag para geração de matriz de confusão
    - output_dir: diretório de saída
    - classifier_type: tipo de classificador
    - out_label: rótulo de saída
    - path_confusion_matrix: caminho para salvar matriz de confusão
    - verbose_level: nível de verbosidade
    - dict: dicionário para armazenar resultados das métricas de utilidade
    - TR_As_aucs: lista para armazenar AUCs dos classificadore TR_As
    

    Retorna:
    - dict: dicionário atualizado com resultados
    - TR_As_aucs: lista atualizada com AUCs
    """
    instance_metrics = ProbabilisticMetrics()
    y_predict_prob=[]
    logging.info(f"TR_As Fold {fold + 1}/{k} results\n")
    ## Itera sobre a lista dos classificadores
    for index, classifier_model in enumerate(list_classifiers):
        ## obtém a predição do classificador em relação aos dados
        if classifier_type[index] == "Perceptron":
            y_predicted_TR_As = classifier_model.predict(x_TR_As, verbose=DEFAULT_VERBOSE_LIST[verbose_level])
            y_predicted_TR_As = np.rint(np.squeeze(y_predicted_TR_As, axis=1))
            y_predict_prob=y_predicted_TR_As
        else:
            y_predicted_TR_As = classifier_model.predict(x_TR_As)
            y_predict_prob=classifier_model.predict_proba(x_TR_As)[::,1]
        y_predicted_TR_As = y_predicted_TR_As.astype(int)

        y_TR_As = y_TR_As.astype(int)
        ##cria uma matrix de confusão
        confusion_matrix_TR_As = confusion_matrix(y_TR_As, y_predicted_TR_As)
        ##Obtem as métricas de utilidade do classificador
        accuracy_TR_As = instance_metrics.get_accuracy(y_TR_As, y_predicted_TR_As)
        precision_TR_As = instance_metrics.get_precision(y_TR_As, y_predicted_TR_As)
        recall_TR_As = instance_metrics.get_recall(y_TR_As, y_predicted_TR_As)
        f1_score_TR_As = instance_metrics.get_f1_score(y_TR_As, y_predicted_TR_As)
        ##Realiza o logging das métricas de fidelidade obtidas
        logging.info(f" Classifier Model: {classifier_type[index]}")
        logging.info(f"   TR_As Fold {fold + 1} - Confusion Matrix:")
        logging.info(confusion_matrix_TR_As)
        logging.info(f"\n   Classifier Metrics:")
        logging.info(f"     TR_As Fold {fold + 1} - Accuracy: " + str(accuracy_TR_As))
        logging.info(f"     TR_As Fold {fold + 1} - Precision: " + str(precision_TR_As))
        logging.info(f"     TR_As Fold {fold + 1} - Recall: " + str(recall_TR_As))
        logging.info(f"     TR_As Fold {fold + 1} - F1 Score: " + str(f1_score_TR_As) + "\n")
        
        ## cria uma lista com os nomes das métricas, será utilizada pelas ferramentas de rastreamento
        ac='TR_As Accuracy '+f' Classifier Model {classifier_type[index]}'
        pc='TR_As Precision '+f' Classifier Model {classifier_type[index]}'
        rr='TR_As Recall '+f' Classifier Model {classifier_type[index]}'
        f1='TR_As f1 Score '+f' Classifier Model {classifier_type[index]}'
        values={ac:accuracy_TR_As,pc:precision_TR_As,rr:recall_TR_As,f1:f1_score_TR_As}
        
        ##Salvamento das métricas no dicionário de métricas
        dict["TR_As accuracy"][classifier_type[index]].append(accuracy_TR_As)
        dict["TR_As precision"][classifier_type[index]].append(precision_TR_As)
        dict["TR_As F1 score"][classifier_type[index]].append(f1_score_TR_As)
        dict["TR_As recall"][classifier_type[index]].append(recall_TR_As)

        ##calculo das métricas da curva de ROC
        fpr, tpr,thresholds=sklearn.metrics.roc_curve(y_TR_As,   y_predict_prob)
        ##plot da figura da curva de ROC
        plt.figure()  
        Path(os.path.join(output_dir, path_confusion_matrix)).mkdir(parents=True, exist_ok=True)
        roc_file = os.path.join(output_dir, path_confusion_matrix,f'Roc_curve_TR_As_{classifier_type[index]}_k{fold + 1}.jpg')
        auc=metrics.roc_auc_score(y_TR_As,   y_predict_prob)
        plt.plot(fpr, tpr)
        plt.savefig(roc_file, bbox_inches="tight")
        ##Salvamento do AUC no dicnário TR_As
        TR_As_aucs[classifier_type[index]].append(auc)
        
        if generate_confusion_matrix:
            ##realiza o plot da figura da matrix de confusão
            plt.figure()
            selected_color_map = plt.colormaps.get_cmap(DEFAULT_COLOR_NAME[(fold + 2) % len(DEFAULT_COLOR_NAME)])
            confusion_matrix_instance = PlotConfusionMatrix()
            confusion_matrix_instance.plot_confusion_matrix(confusion_matrix_TR_As, out_label, selected_color_map)
            Path(os.path.join(output_dir, path_confusion_matrix)).mkdir(parents=True, exist_ok=True)
            matrix_file = os.path.join(output_dir, path_confusion_matrix,
                                       f'CM_TR_As_{classifier_type[index]}_k{fold + 1}.jpg')
            plt.savefig(matrix_file, bbox_inches='tight')
            ## opção para salvamento das figuras geradas e as métricas obtidas, na  ferramenta de rastreamento Aimstack 
            if USE_AIM:
               aim_run.track(values)
               plt.savefig(matrix_file, bbox_inches='tight')
               aim_image = Image(matrix_file)
               aim_run.track(value=aim_image,name= f'CM_TR_As_{classifier_type[index]}_k{fold + 1}')
               ## opção para salvamento das figuras geradas e as métricas obtidas, na  ferramenta de rastreamento Mlflow
            if USE_MLFLOW:
                  mlflow.log_metrics(values,step=fold+1)
                  mlflow.log_artifact(matrix_file, 'images')
            ## opção para salvamento das figuras geradas e as métricas obtidas, na  ferramenta de rastreamento TensorBoard
            if USE_TENSORBOARD:
               with file_writer.as_default():
                  cm_image = plot_to_image(matrix_file)
                  tf.summary.image(f'CM_TS_Ar_{classifier_type[index]}_k{fold + 1}', cm_image,step=fold+1)
                  tf.summary.scalar(ac, data=accuracy_TR_As, step=fold+1)
                  tf.summary.scalar(pc, data=precision_TR_As, step=fold+1)
                  tf.summary.scalar(rr, data=recall_TR_As, step=fold+1)
                  tf.summary.scalar(f1, data=f1_score_TR_As, step=fold+1)



def evaluate_TS_Ar_data(list_classifiers, x_TS_Ar, y_TS_Ar, fold, k, generate_confusion_matrix, output_dir,
                       classifier_type, out_label, path_confusion_matrix, verbose_level,dict,TS_Ar_aucs):
    """
    Avalia os do classificador TS_Ar(Treinado com dados sintéticos/ Avaliado com dados reais) 

    Parâmetros:
    - list_classifiers: lista de classificadores
    - x_TS_Ar: dados utilizada para evaluar os classificadores TS_Ar
    - y_TS_Ar: rótulos dos dados de avaliação
    - fold: índice da iteração atual
    - k: índice da subdivisão atual
    - generate_confusion_matrix: flag para geração de matriz de confusão
    - output_dir: diretório de saída
    - classifier_type: tipo de classificador
    - out_label: rótulo de saída
    - path_confusion_matrix: caminho para salvar matriz de confusão
    - verbose_level: nível de verbosidade
    - dict: dicionário para armazenar resultados das métricas de utilidade
    - TS_Ar_aucs: lista para armazenar AUCs dos classificadores TS_Ar


    Retorna:
    - dict: dicionário atualizado com resultados
    - TS_Ar_aucs: lista atualizada com AUCs
  
    """
    logging.info(f"TS_Ar Fold {fold + 1}/{k} results")
    
    instance_metrics = ProbabilisticMetrics()
    y_predict_prob=[]
    ## Itera sobre a lista dos classificadores
    for index, classifier_model in enumerate(list_classifiers):
        ## obtém a predição do classificador em relação aos dados
        if classifier_type[index] == "Perceptron":
            y_predicted_TS_Ar = classifier_model.predict(x_TS_Ar, verbose=DEFAULT_VERBOSE_LIST[verbose_level])
            y_predicted_TS_Ar = np.rint(np.squeeze(y_predicted_TS_Ar, axis=1))
            y_predict_prob=y_predicted_TS_Ar
        else:
            print(classifier_type[index])
            y_predicted_TS_Ar = classifier_model.predict(x_TS_Ar)
            y_predict_prob=classifier_model.predict_proba(x_TS_Ar)[::,1]

        y_predicted_TS_Ar = y_predicted_TS_Ar.astype(int)
        y_sample_TS_Ar = y_TS_Ar.astype(int)
        ##cria uma matrix de confusão
        confusion_matrix_TS_Ar = confusion_matrix(y_sample_TS_Ar, y_predicted_TS_Ar)
        ##Obtem as métricas de utilidade do classificador
        accuracy_TS_Ar = instance_metrics.get_accuracy(y_sample_TS_Ar, y_predicted_TS_Ar)
        precision_TS_Ar = instance_metrics.get_precision(y_sample_TS_Ar, y_predicted_TS_Ar)
        recall_TS_Ar = instance_metrics.get_recall(y_sample_TS_Ar, y_predicted_TS_Ar)
        f1_TS_Ar = instance_metrics.get_f1_score(y_sample_TS_Ar, y_predicted_TS_Ar)
        ##Realiza o logging das métricas de fidelidade obtidas
        logging.info(f" Classifier Model: {classifier_type[index]}")
        logging.info(f"   TS_Ar Fold {fold + 1} - Confusion Matrix:")
        logging.info(confusion_matrix_TS_Ar)
        logging.info(f"\n   Classifier Metrics:")
        logging.info(f"     TS_Ar Fold {fold + 1} - Accuracy: " + str(accuracy_TS_Ar))
        logging.info(f"     TS_Ar Fold {fold + 1} - Precision: " + str(precision_TS_Ar))
        logging.info(f"     TS_Ar Fold {fold + 1} - Recall: " + str(recall_TS_Ar))
        logging.info(f"     TS_Ar Fold {fold + 1} - F1 Score: " + str(f1_TS_Ar) + "\n")
        logging.info("")

        ## cria uma lista com os nomes das métricas, será utilizada pelas ferramentas de rastreamento
        ac='TS_Ar Accuracy '+f' Classifier Model {classifier_type[index]}'
        pc='TS_Ar precision '+f' Classifier Model {classifier_type[index]}'
        rr='TS_Ar Recall '+f' Classifier Model {classifier_type[index]}'
        f1='TS_Ar f1 Score '+f' Classifier Model {classifier_type[index]}'
        values={ac:accuracy_TS_Ar,pc:precision_TS_Ar,rr:recall_TS_Ar,f1:f1_TS_Ar}

        ##Salvamento das métricas no dicionário de métricas
        dict["TS_Ar accuracy"][classifier_type[index]].append(accuracy_TS_Ar)
        dict["TS_Ar precision"][classifier_type[index]].append(precision_TS_Ar)
        dict["TS_Ar F1 score"][classifier_type[index]].append(f1_TS_Ar)
        dict["TS_Ar recall"][classifier_type[index]].append(recall_TS_Ar)

        ##calculo das métricas da curva de ROC
        fpr, tpr,thresholds=metrics.roc_curve(y_sample_TS_Ar, y_predict_prob)
        ##plot da figura da curva de ROC
        plt.figure()  
        Path(os.path.join(output_dir, path_confusion_matrix)).mkdir(parents=True, exist_ok=True)
        roc_file = os.path.join(output_dir, path_confusion_matrix,f'Roc_curve_TS_Ar_{classifier_type[index]}_k{fold + 1}.jpg')
        auc=metrics.roc_auc_score(y_sample_TS_Ar,  y_predict_prob)
        plt.plot(fpr, tpr)
        plt.savefig(roc_file, bbox_inches="tight")
        ##Salvamento do AUC no dicnário TR_As
        TS_Ar_aucs[classifier_type[index]].append(auc)
        if generate_confusion_matrix:
            ##realiza o plot da figura da matrix de confusão
            plt.figure()
            selected_color_map = plt.colormaps.get_cmap(DEFAULT_COLOR_NAME[(fold + 2) % len(DEFAULT_COLOR_NAME)])
            confusion_matrix_instance = PlotConfusionMatrix()
            confusion_matrix_instance.plot_confusion_matrix(confusion_matrix_TS_Ar, out_label, selected_color_map)
            Path(os.path.join(output_dir, path_confusion_matrix)).mkdir(parents=True, exist_ok=True)
            matrix_file = os.path.join(output_dir, path_confusion_matrix,
                                       f'CM_TS_Ar_{classifier_type[index]}_k{fold + 1}.jpg')
            matrix=plt.savefig(matrix_file, bbox_inches='tight')
            cm_image = plot_to_image(matrix_file)
            ##  opção para salvamento das figuras geradas e as métricas obtidas, na  ferramenta de rastreamento TensorBoard
            if USE_TENSORBOARD:
               with file_writer.as_default():
                  tf.summary.image(f'CM_TS_Ar_{classifier_type[index]}_k{fold + 1}', cm_image,step=fold+1)
                  tf.summary.scalar(ac, data=accuracy_TS_Ar, step=fold+1)
                  tf.summary.scalar(pc, data=precision_TS_Ar, step=fold+1)
                  tf.summary.scalar(rr, data=recall_TS_Ar, step=fold+1)
                  tf.summary.scalar(f1, data=f1_TS_Ar, step=fold+1)
            ## ## opção para salvamento das figuras geradas e as métricas obtidas, na  ferramenta de rastreamento Aimstack 
            if USE_AIM:
               aim_run.track(values)
               plt.savefig(matrix_file, bbox_inches='tight')
               aim_image = Image(matrix_file)
               aim_run.track(value=aim_image, name=f'CM_TS_Ar_{classifier_type[index]}_k{fold + 1}')
            ## opção para salvamento das figuras geradas e as métricas obtidas, na  ferramenta de rastreamento Mlflow
            if USE_MLFLOW:
                  mlflow.log_metrics(values,step=fold+1)
                  mlflow.log_artifact(matrix_file, 'images')

def p_value_test (TS_Ar_label,TR_As_label,type_of_metric,classifier_type):
    """
    A função calcula o valor-p (p-value) utilizando o teste de Wilcoxon para amostras pareadas das métricas de utilidade dos classificadores.

    Parâmetros:
    -TS_Ar_label : Dicionário contendo os dados dos classificadores TS_Ar, onde as chaves são os tipos de classificadores 
                       e os valores são listas ou arrays com os dados.
    -TR_As_label : Dicionário contendo os dados dos classificadores TR_As, onde as chaves são os tipos de 
                        classificadores e os valores são listas ou arrays com os dados.
    -type_of_metric : Uma string que representa o tipo de métrica que está sendo testada.
    -classifier_type : Uma string que representa o tipo de classificador cujos dados serão comparados.

    Retorna:
    -p_value: O valor-p resultante do teste de Wilcoxon.
    """
    TS_Ar=[]
    TR_As=[]
    estaticts=0
    p_value=0
    TS_Ar=np.asfarray(TS_Ar_label[classifier_type])
    TR_As=np.asfarray(TR_As_label[classifier_type])   
    estaticts, p_value = stats.wilcoxon(TS_Ar,TR_As,alternative='two-sided')
    logging.info("  P_value of {}  {} and stats {} of {} \n".format(type_of_metric,p_value,estaticts,classifier_type))
    return p_value

def show_and_export_results(dict_similarity, classifier_type, output_dir, title_output_label, dict_metrics, dict_TR_As_auc, dict_TS_Ar_auc):
    """
    Função para demonstrar e exportar resultados de métricas de fidelidade e utilidade, além da chamada de criação dos plots destas métricas.
    
    Parâmetros:
        -dict_similarity : Dicionário contendo listas de métricas de fidelidade entre os dados.
        -classifier_type : Lista de classificadores.
        -output_dir (: Diretório de saída para salvar os resultados.
        -title_output_label : Título do rótulo de saída.
        -dict_metrics : Dicionário contendo listas de métricas (ex. precisão, recall, etc.).
        -dict_TR_As_auc : Dicionário contendo AUC para os classificadores TR_As.
        -dict_TS_Ar_auc : Dicionário contendo AUC para os classificadores TS_Ar.
    """
    
    ## Inicializa as classes para plotar métricas
    plot_classifier_metrics = PlotClassificationMetrics()
    plot_fidelity_metrics = PlotFidelityeMetrics()
    
    ## Itera sobre os classificadores
    for index in range(len(classifier_type)):
        
        ## Logging dos resultados dos classificadores TR_As
        logging.info("Overall TR_As Results: Classifier {}\n".format(classifier_type[index]))
        logging.info("  TR_As List of Accuracies: {} ".format(dict_metrics["TR_As accuracy"][classifier_type[index]]))
        logging.info("  TR_As List of Precisions: {} ".format(dict_metrics["TR_As precision"][classifier_type[index]]))
        logging.info("  TR_As List of Recalls: {} ".format(dict_metrics["TR_As recall"][classifier_type[index]]))
        logging.info("  TR_As List of F1-scores: {} ".format(dict_metrics["TR_As F1 score"][classifier_type[index]]))
      
        logging.info("  TR_As list AUC: {} ".format((dict_TR_As_auc[classifier_type[index]])))
        logging.info("  TR_As Mean Accuracy: {} ".format(np.mean(dict_metrics["TR_As accuracy"][classifier_type[index]])))
        logging.info("  TR_As Mean Precision: {} ".format(np.mean(dict_metrics["TR_As precision"][classifier_type[index]])))
        logging.info("  TR_As Mean Recall: {} ".format(np.mean(dict_metrics["TR_As recall"][classifier_type[index]])))
        logging.info("  TR_As Mean F1 Score: {} ".format(np.mean(dict_metrics["TR_As F1 score"][classifier_type[index]])))
        logging.info("  TR_As Mean AUC: {} ".format(np.mean(dict_TR_As_auc[classifier_type[index]])))
      
        logging.info("  TR_As Standard Deviation of Accuracy: {} ".format(np.std(dict_metrics["TR_As accuracy"][classifier_type[index]])))
        logging.info("  TR_As Standard Deviation of Precision: {} ".format(np.std(dict_metrics["TR_As precision"][classifier_type[index]])))
        logging.info("  TR_As Standard Deviation of Recall: {} ".format(np.std(dict_metrics["TR_As recall"][classifier_type[index]])))
        logging.info("  TR_As Standard Deviation of F1 Score: {} \n".format(np.std(dict_metrics["TR_As F1 score"][classifier_type[index]])))
        logging.info("  TR_As Standard Deviation of AUC: {} ".format(np.std(dict_TR_As_auc[classifier_type[index]])))
        ## Nome do plot
        plot_filename = os.path.join(output_dir, f'{classifier_type[index]}_TS_Ar_(Treinado com sintético,avalia com dados reais)_.pdf')

        ## Plota e salva as métricas dos classificadores  TR_As
        plot_classifier_metrics.plot_classifier_metrics(
            classifier_type[index], 
            dict_metrics["TR_As accuracy"][classifier_type[index]],
            dict_metrics["TR_As precision"][classifier_type[index]], 
            dict_metrics["TR_As recall"][classifier_type[index]], 
            dict_metrics["TR_As F1 score"][classifier_type[index]], 
            plot_filename,
            f'{title_output_label}_TR_As', "TR_As"
        )

        ## Define os nomes das métricas TR_As para as ferramentas de rastreamento
        ac = 'TR_As Mean Accuracy ' + f' Classifier Model {classifier_type[index]}'
        pc = 'TR_As Mean Precision ' + f' Classifier Model {classifier_type[index]}'
        rr = 'TR_As Mean Recall ' + f' Classifier Model {classifier_type[index]}'
        f1 = 'TR_As Mean F1 Score ' + f' Classifier Model {classifier_type[index]}'
        pct = 'TR_As Standard Deviation of Precision ' + f'Classifier Model {classifier_type[index]}'
        act = 'TR_As Standard Deviation of Accuracy ' + f' Classifier Model {classifier_type[index]}'
        rrt = 'TR_As Standard Deviation of Recall ' + f' Classifier Model {classifier_type[index]}'
        f1t = 'TR_As Standard Deviation of F1 Score ' + f' Classifier Model {classifier_type[index]}'
        
        ## Rastreamento das métricas utilizando Aimstack
        if USE_AIM:
            values = {
                ac: np.mean(dict_metrics["TR_As accuracy"][classifier_type[index]]),
                pc: np.mean(dict_metrics["TR_As precision"][classifier_type[index]]),
                rr: np.mean(dict_metrics["TR_As recall"][classifier_type[index]]),
                f1: np.mean(dict_metrics["TR_As F1 score"][classifier_type[index]]),
                act: np.std(dict_metrics["TR_As accuracy"][classifier_type[index]]),
                pct: np.std(dict_metrics["TR_As precision"][classifier_type[index]]),
                rrt: np.std(dict_metrics["TR_As recall"][classifier_type[index]]),
                f1t: np.std(dict_metrics["TR_As F1 score"][classifier_type[index]]),
            }

            aim_run.track(values, context={'data': 'TR_As', 'classifier_type': classifier_type[index]})
               
        ## Rastreamento das métricas utilizando Mlflow
        if USE_MLFLOW:
            values = {
                ac: np.mean(dict_metrics["TR_As accuracy"][classifier_type[index]]),
                pc: np.mean(dict_metrics["TR_As precision"][classifier_type[index]]),
                rr: np.mean(dict_metrics["TR_As recall"][classifier_type[index]]),
                f1: np.mean(dict_metrics["TR_As F1 score"][classifier_type[index]]),
                act: np.std(dict_metrics["TR_As accuracy"][classifier_type[index]]),
                pct: np.std(dict_metrics["TR_As precision"][classifier_type[index]]),
                rrt: np.std(dict_metrics["TR_As recall"][classifier_type[index]]),
                f1t: np.std(dict_metrics["TR_As F1 score"][classifier_type[index]]),
            }
            mlflow.log_metrics(values)

        ## Rastreamento das métricas utilizando TensorBoard
        if USE_TENSORBOARD:
            with file_writer.as_default():
                tf.summary.scalar(ac, data=np.mean(dict_metrics["TR_As accuracy"][classifier_type[index]]), step=0)
                tf.summary.scalar(pc, data=np.mean(dict_metrics["TR_As precision"][classifier_type[index]]), step=0)
                tf.summary.scalar(rr, data=np.mean(dict_metrics["TR_As recall"][classifier_type[index]]), step=0)
                tf.summary.scalar(f1, data=np.mean(dict_metrics["TR_As F1 score"][classifier_type[index]]), step=0)
                tf.summary.scalar(act, data=np.std(dict_metrics["TR_As accuracy"][classifier_type[index]]), step=0)
                tf.summary.scalar(pct, data=np.std(dict_metrics["TR_As precision"][classifier_type[index]]), step=0)
                tf.summary.scalar(rrt, data=np.std(dict_metrics["TR_As recall"][classifier_type[index]]), step=0)
                tf.summary.scalar(f1t, data=np.std(dict_metrics["TR_As F1 score"][classifier_type[index]]), step=0)

        ## logging  dos resultados dos classificadores TS_Ar
        logging.info("Overall TS_Ar Results: {}\n".format(classifier_type[index]))
        logging.info("  TS_Ar List of Accuracies: {} ".format(dict_metrics["TS_Ar accuracy"][classifier_type[index]]))
        logging.info("  TS_Ar List of Precisions: {} ".format(dict_metrics["TS_Ar precision"][classifier_type[index]]))
        logging.info("  TS_Ar List of Recalls: {} ".format(dict_metrics["TS_Ar recall"][classifier_type[index]]))
        logging.info("  TS_Ar List of F1-scores: {} ".format(dict_metrics["TS_Ar F1 score"][classifier_type[index]]))
        logging.info("  TS_Ar list AUC: {} ".format((dict_TS_Ar_auc[classifier_type[index]])))
       
        logging.info("  TS_Ar Mean Accuracy: {} ".format(np.mean(dict_metrics["TS_Ar accuracy"][classifier_type[index]])))
        logging.info("  TS_Ar Mean Precision: {} ".format(np.mean(dict_metrics["TS_Ar precision"][classifier_type[index]])))
        logging.info("  TS_Ar Mean Recall: {} ".format(np.mean(dict_metrics["TS_Ar recall"][classifier_type[index]])))
        logging.info("  TS_Ar Mean F1 Score: {} ".format(np.mean(dict_metrics["TS_Ar F1 score"][classifier_type[index]])))
        
        logging.info("  TS_Ar Mean AUC: {} ".format(np.mean(dict_TS_Ar_auc[classifier_type[index]])))
        logging.info("  TS_Ar Standard Deviation of Accuracy: {} ".format(np.std(dict_metrics["TS_Ar accuracy"][classifier_type[index]])))
        logging.info("  TS_Ar Standard Deviation of Precision: {} ".format(np.std(dict_metrics["TS_Ar precision"][classifier_type[index]])))
        logging.info("  TS_Ar Standard Deviation of Recall: {} ".format(np.std(dict_metrics["TS_Ar recall"][classifier_type[index]])))
        logging.info("  TS_Ar Standard Deviation of F1 Score: {} \n".format(np.std(dict_metrics["TS_Ar F1 score"][classifier_type[index]])))
        logging.info("  TS_Ar Standard Deviation of AUC: {} ".format(np.std(dict_TS_Ar_auc[classifier_type[index]])))
        ## Nome do plot
        plot_filename = os.path.join(output_dir, f'{classifier_type[index]}_treiando_com_sint_testado_com_TS_Ar.pdf')

        ## Plota e salva as métricas dos classificadores  TS_Ar
        plot_classifier_metrics.plot_classifier_metrics(
            classifier_type[index], 
            dict_metrics["TS_Ar accuracy"][classifier_type[index]],
            dict_metrics["TS_Ar precision"][classifier_type[index]],
            dict_metrics["TS_Ar recall"][classifier_type[index]],
            dict_metrics["TS_Ar F1 score"][classifier_type[index]],
            plot_filename,
            f'{title_output_label}_TS_Ar', "TS_Ar"
        )

        ## Define os nomes das métricas TS_Ar para as ferramentas de rastreamento
        ac = 'TS_Ar Mean Accuracy ' + f' Classifier Model {classifier_type[index]}'
        pc = 'TS_Ar Mean Precision ' + f' Classifier Model {classifier_type[index]}'
        rr = 'TS_Ar Mean Recall ' + f' Classifier Model {classifier_type[index]}'
        f1 = 'TS_Ar Mean f1 Score ' + f' Classifier Model {classifier_type[index]}'
        pct = 'TS_Ar Standart Deviation of PreCision ' + f'Classifier Model {classifier_type[index]}'
        act = 'TS_Ar Standard Deviation of Accuracy ' + f' Classifier Model {classifier_type[index]}'
        rrt = 'TS_Ar Standard Deviation of Recall ' + f' Classifier Model {classifier_type[index]}'
        f1t = 'TS_Ar Standard Deviation of F1 Score ' + f' Classifier Model {classifier_type[index]}'

        ## Rastreamento das métricas utilizando Aimstack
        if USE_AIM:
            aim_run.track(values, context={'data': 'TS_Ar', 'classifir_type': classifier_type[index]})

        ## Rastreamento das métricas utilizando Mlflow
        if USE_MLFLOW:
            mlflow.log_metrics(values)

        ## Rastreamento das métricas utilizando TensorBoard
        if USE_TENSORBOARD:
            with file_writer.as_default():
                tf.summary.scalar(ac, data=np.mean(dict_metrics["TS_Ar accuracy"][classifier_type[index]]), step=0)
                tf.summary.scalar(pc, data=np.mean(dict_metrics["TS_Ar precision"][classifier_type[index]]), step=0)
                tf.summary.scalar(rr, data=np.mean(dict_metrics["TS_Ar recall"][classifier_type[index]]), step=0)
                tf.summary.scalar(f1, data=np.mean(dict_metrics["TS_Ar F1 score"][classifier_type[index]]), step=0)
                tf.summary.scalar(act, data=np.std(dict_metrics["TS_Ar accuracy"][classifier_type[index]]), step=0)
                tf.summary.scalar(pct, data=np.std(dict_metrics["TS_Ar precision"][classifier_type[index]]), step=0)
                tf.summary.scalar(rrt, data=np.std(dict_metrics["TS_Ar recall"][classifier_type[index]]), step=0)
                tf.summary.scalar(f1t, data=np.std(dict_metrics["TS_Ar F1 score"][classifier_type[index]]), step=0)

    ## Logging das métricas fidelidade entre os dados sintéticos e reais
    comparative_metrics = ['Mean Squared Error', 'Cosine Similarity', 'Max Mean Discrepancy']
    comparative_lists = ["list_mean_squared_error", "list_cosine_similarity", "list_maximum_mean_discrepancy"]
    logging.info(f"Comparative Metrics:")
    for metric, comparative_list in zip(comparative_metrics, comparative_lists):
        logging.info("\t{}".format(metric))
        for i in ["false", "positive"]:
            logging.info("\t\t {}".format(i))
            logging.info("\t\t{} - List     : {}".format(metric, dict_similarity[comparative_list][i]))
            logging.info("\t\t{} - Mean    : {}".format(metric, np.mean(dict_similarity[comparative_list][i])))
            logging.info("\t\t{} - Std. Dev.  : {}\n".format(metric, np.std(dict_similarity[comparative_list][i])))
    
    ## Testes de valor-p
    for index in range(len(classifier_type)):
        p_value_test(dict_metrics["TS_Ar accuracy"], dict_metrics["TR_As accuracy"], "accuracy", classifier_type[index])
        p_value_test(dict_metrics["TS_Ar precision"], dict_metrics["TR_As precision"], "precision", classifier_type[index])
        p_value_test(dict_metrics["TS_Ar F1 score"], dict_metrics["TR_As F1 score"], "F1 score", classifier_type[index])
        p_value_test(dict_metrics["TS_Ar recall"], dict_metrics["TR_As recall"], "recall", classifier_type[index])
        p_value_test(dict_TS_Ar_auc, dict_TR_As_auc, "auc", classifier_type[index])
        plot_filename = os.path.join(output_dir, f'{classifier_type[index]}_p_values.pdf')

    ## Plota e salva as métricas de fidelidade geradas
    plot_filename1 = os.path.join(output_dir, f'Comparison_TS_Ar_TR_As_positive.jpg')
    plot_filename2 = os.path.join(output_dir, f'Comparison_TS_Ar_TR_As_false.jpg')

    plot_fidelity_metrics.plot_fidelity_metrics(
        dict_similarity["list_mean_squared_error"]["positive"],
        dict_similarity["list_cosine_similarity"]["positive"],
        dict_similarity["list_maximum_mean_discrepancy"]["positive"],
        plot_filename1,
        f'{title_output_label}'
    )
    plot_fidelity_metrics.plot_fidelity_metrics(
        dict_similarity["list_mean_squared_error"]["false"],
        dict_similarity["list_cosine_similarity"]["false"],
        dict_similarity["list_maximum_mean_discrepancy"]["false"],
        plot_filename2,
        f'{title_output_label}'
    )
    
    ## Rastreamento das imagens utilizando Aimstack
    if USE_AIM:
        aim_image = Image(plot_filename1)
        aim_image2 = Image(plot_filename2)
        aim_run.track(value=aim_image, name=plot_filename1, context={'classifier_type': classifier_type[index]})
        aim_run.track(value=aim_image2, name=plot_filename2, context={'classifier_type': classifier_type[index]})
    
    ## Rastreamento das imagens utilizando TensorBoard
    if USE_TENSORBOARD:
        with file_writer.as_default():
            cm_image = plot_to_image(plot_filename)
            tf.summary.image('Comparison_TS_Ar_TR_As.jpg', cm_image, step=0)
    
    ## Rastreamento das imagens utilizando Mlflow
    if USE_MLFLOW:
        mlflow.log_artifact(plot_filename1, 'images')
        mlflow.log_artifact(plot_filename2, 'images')



def get_adversarial_model(latent_dim, input_data_shape, activation_function, initializer_mean, initializer_deviation,
                          dropout_decay_rate_g, dropout_decay_rate_d, last_layer_activation, dense_layer_sizes_g,
                          dense_layer_sizes_d, dataset_type, training_algorithm, latent_mean_distribution,
                          latent_stander_deviation):
    """
    Função utilizada para a chamada do método de instanciamento da rede adversarial cGAN.
    
    Parâmetros:
        -latent_dim:  Dimensão do espaço latente para treinamento cGAN.
        -input_data_shape: Formato dos dados de entrada: Ex Float32
        -activation_function: Função de ativação da cGAN. 
        -initializer_mean: Valor central da distribuição gaussiana do inicializador.
        -initializer_deviation: Desvio padrão da distribuição gaussiana do inicializador.
        -dropout_decay_rate_g: Taxa de decaimento do dropout do gerador da cGAN.
        -dropout_decay_rate_d: Taxa de decaimento do dropout do discriminador da cGAN.
        -last_layer_activation: 
        -dense_layer_sizes_g:  Valores das camadas densas do gerador.
        -dense_layer_sizes_d:  Valores das camadas densas do discriminador
        -training_algorithm:Algoritmo de treinamento para cGAN.
        -latent_mean_distribution:Valor central da distribuição da camada latente.
        -latent_stander_deviation:  Desvio padrão da distribuição da camada latente.

    Retorna:
     -adversarial_model: Modelo adversarial instanciado e configurado
    """
    
    ## Criação de uma instância da Conditional GAN com os parâmetros especificados
    instance_models = ConditionalGAN(latent_dim, input_data_shape, activation_function, initializer_mean,
                                 initializer_deviation, dropout_decay_rate_g, dropout_decay_rate_d,
                                 last_layer_activation, dense_layer_sizes_g, dense_layer_sizes_d, dataset_type)

    ## Obtenção dos modelos do gerador e do discriminador da instância criada
    generator_model = instance_models.get_generator()
    discriminator_model = instance_models.get_discriminator()

    ## Criação do modelo adversarial combinando o gerador e o discriminador
    adversarial_model = AdversarialModel(generator_model, discriminator_model, latent_dimension=latent_dim,
                                     latent_mean_distribution=latent_mean_distribution,
                                     latent_stander_deviation=latent_stander_deviation)

    ## Configuração dos otimizadores para o gerador e o discriminador
    optimizer_generator = adversarial_model.get_optimizer(training_algorithm)
    optimizer_discriminator = adversarial_model.get_optimizer(training_algorithm)

    ## Definição das funções de perda para o gerador e o discriminador
    loss_generator = BinaryCrossentropy()
    loss_discriminator = BinaryCrossentropy()

    adversarial_model.compile(optimizer_generator, optimizer_discriminator, loss_generator, loss_discriminator,  loss=keras.losses.SparseCategoricalCrossentropy(),)
    return adversarial_model

def show_model(latent_dim, input_data_shape, activation_function, initializer_mean,
               initializer_deviation, dropout_decay_rate_g, dropout_decay_rate_d,
               last_layer_activation, dense_layer_sizes_g, dense_layer_sizes_d,
               dataset_type, verbose_level):
    """
    A função imprime os parametros da rede adversarial no log de saida.
    
    Parâmetros:
        -latent_dim:  Dimensão do espaço latente para treinamento cGAN.
        -input_data_shape: Formato dos dados de entrada (número de linhas e colunas).
        -activation_function: Função de ativação da cGAN. 
        -initializer_mean: Valor central da distribuição gaussiana do inicializador.
        -initializer_deviation: Desvio padrão da distribuição gaussiana do inicializador.
        -dropout_decay_rate_g: Taxa de decaimento do dropout do gerador da cGAN.
        -dropout_decay_rate_d: Taxa de decaimento do dropout do discriminador da cGAN.
        -last_layer_activation:  Função de ativação para a última camada do gerador
        -dense_layer_sizes_g:  Valores das camadas densas do gerador.
        -dense_layer_sizes_d:  Valores das camadas densas do discriminador
        -training_algorithm:Algoritmo de treinamento para cGAN.
        -latent_mean_distribution:Valor central da distribuição da camada latente.
        -latent_stander_deviation:  Desvio padrão da distribuição da camada latente.

    """   
    show_model_instance = ConditionalGAN(latent_dim, input_data_shape, activation_function, initializer_mean,
                                         initializer_deviation, dropout_decay_rate_g, dropout_decay_rate_d,
                                         last_layer_activation, dense_layer_sizes_g, dense_layer_sizes_d,
                                         dataset_type)
def run_experiment(dataset, input_data_shape, k, classifier_list, output_dir, batch_size, training_algorithm,
                   number_epochs, latent_dim, activation_function, dropout_decay_rate_g, dropout_decay_rate_d,
                   dense_layer_sizes_g=None, dense_layer_sizes_d=None, dataset_type=None, title_output=None,
                   initializer_mean=None, initializer_deviation=None,
                   last_layer_activation=DEFAULT_CONDITIONAL_LAST_ACTIVATION_LAYER, save_models=False,
                   path_confusion_matrix=None, path_curve_loss=None, verbose_level=None,
                   latent_mean_distribution=None, latent_stander_deviation=None, num_samples_class_malware=None, num_samples_class_benign=None):
    """
   Responsável pelo principal fluxo de execução da cGAN, realiza as chamada das funções de:
       - Treino da cGAN.
       - Geração de dados sintéticos.
       - Comparação de dados.
       - Cálculo de métricas de utilidade e fidelidade.
       - Export de resultados


    Parâmetros:
        - dataset: Dataset de entrada no formato CSV.
        - input_data_shape: Formato dos dados de entrada (número de linhas e colunas).
        - k: Número de folds a serem realizados durante o experimento (K-fold cross-validation).
        - classifier_list: Lista dos classificadores a serem utilizados para a avaliação da performance da cGAN.
        - output_dir: Diretório de saída para salvar os resultados e modelos.
        - batch_size: Tamanho do batch para treinamento.
        - training_algorithm: Algoritmo de treinamento a ser utilizado.
        - number_epochs: Número de épocas para treinamento.
        - latent_dim: Dimensão do espaço latente.
        - activation_function: Função de ativação para as camadas internas.
        - dropout_decay_rate_g: Taxa de dropout para o gerador.
        - dropout_decay_rate_d: Taxa de dropout para o discriminador.
        - dense_layer_sizes_g: Tamanhos das camadas densas do gerador.
        - dense_layer_sizes_d: Tamanhos das camadas densas do discriminador.
        - dataset_type: Tipo de dados do dataset.
        - title_output: Título para a saída dos resultados.
        - initializer_mean: Média para inicialização dos pesos das camadas.
        - initializer_deviation: Desvio padrão para inicialização dos pesos das camadas.
        - last_layer_activation: Função de ativação para a última camada do gerador.
        - save_models: Se True, salva os modelos treinados.
        - path_confusion_matrix: Caminho para salvar a matriz de confusão.
        - path_curve_loss: Caminho para salvar a curva de perda.
        - verbose_level: Nível de verbosidade para logging.
        - latent_mean_distribution : Média da distribuição latente.
        - latent_stander_deviation : Desvio padrão da distribuição latente.
        - num_samples_class_malware: Número de amostras para a classe malware.
        - num_samples_class_benign: Número de amostras para a classe benigna.
    """

    ## Mostrar a configuração do modelo
    show_model(latent_dim, input_data_shape, activation_function, initializer_mean,
               initializer_deviation, dropout_decay_rate_g, dropout_decay_rate_d,
               last_layer_activation, dense_layer_sizes_g, dense_layer_sizes_d,
               dataset_type, verbose_level)

    ## Criação do KFold estratificado
    stratified = StratifiedKFold(n_splits=k, shuffle=True)

    ## Inicialização dos dicionários para armazenar as métricas de similaridade e desempenho dos classificadores
    dict_similarity = {
        "list_mean_squared_error": {"positive": [], "false": []},
        "list_cosine_similarity": {"positive": [], "false": []},
        "list_kl_divergence": {"positive": [], "false": []},
        "list_maximum_mean_discrepancy": {"positive": [], "false": []}
    }
    ## Definição dos dicionários utilizados para armazenar as métricas de utilidade
    dict_TR_As_auc={"RandomForest":[],"AdaBoost":[],"DecisionTree":[], "Perceptron":[],"SupportVectorMachine":[],"SGDRegressor":[],"XGboost":[]}
    dict_TS_Ar_auc={"RandomForest":[],"AdaBoost":[],"DecisionTree":[], "Perceptron":[],"SupportVectorMachine":[],"SGDRegressor":[],"XGboost":[]}
    dict_metrics={"TS_Ar accuracy":{"RandomForest":[],"AdaBoost":[],"DecisionTree":[], "Perceptron":[],"SupportVectorMachine":[],"SGDRegressor":[],"XGboost":[]},
         "TR_As accuracy":{"RandomForest":[],"AdaBoost":[],"DecisionTree":[], "Perceptron":[],"SupportVectorMachine":[],"SGDRegressor":[],"XGboost":[]},
         "TS_Ar precision":{"RandomForest":[],"AdaBoost":[],"DecisionTree":[], "Perceptron":[],"SupportVectorMachine":[],"SGDRegressor":[],"XGboost":[]},
         "TR_As precision":{"RandomForest":[],"AdaBoost":[],"DecisionTree":[], "Perceptron":[],"SupportVectorMachine":[],"SGDRegressor":[],"XGboost":[]},
         "TS_Ar F1 score":{"RandomForest":[],"AdaBoost":[],"DecisionTree":[], "Perceptron":[],"SupportVectorMachine":[],"SGDRegressor":[],"XGboost":[]},
         "TR_As F1 score":{"RandomForest":[],"AdaBoost":[],"DecisionTree":[],"Perceptron":[],"SupportVectorMachine":[],"SGDRegressor":[],"XGboost":[]},
         "TS_Ar recall":{"RandomForest":[],"AdaBoost":[],"DecisionTree":[], "Perceptron":[],"SupportVectorMachine":[],"SGDRegressor":[],"XGboost":[]},
         "TR_As recall":{"RandomForest":[],"AdaBoost":[],"DecisionTree":[], "Perceptron":[],"SupportVectorMachine":[],"SGDRegressor":[],"XGboost":[]}}
    ## Iterar sobre cada fold do KFold estratificado
    for i, (train_index, test_index) in enumerate(stratified.split(dataset.iloc[:, :-1], dataset.iloc[:, -1])):
        ## Obter o modelo adversarial configurado
        adversarial_model = get_adversarial_model(latent_dim, input_data_shape, activation_function, initializer_mean,
                                                  initializer_deviation, dropout_decay_rate_g, dropout_decay_rate_d,
                                                  last_layer_activation, dense_layer_sizes_g, dense_layer_sizes_d,
                                                  dataset_type, training_algorithm, latent_mean_distribution,
                                                  latent_stander_deviation)

        ## Instâncias dos classificadores
        instance_classifier_TR_As = Classifiers()
        instance_classifier_TS_Ar = Classifiers()

        ## Preparação dos dados de treino e teste
        x_training = np.array(dataset.iloc[train_index, :-1].values, dtype=dataset_type)
        x_test = np.array(dataset.iloc[test_index, :-1].values, dtype=dataset_type)
        y_training = np.array(dataset.iloc[train_index, -1].values, dtype=dataset_type)
        y_test = np.array(dataset.iloc[test_index, -1].values, dtype=dataset_type)
        x_training = x_training[:len(x_training) - (len(x_training) % batch_size), :]
        y_training = y_training[:len(y_training) - (len(y_training) % batch_size)]

        logging.info("Iniciando treinamento do modelo adversarial.")

        ## Treinamento do modelo adversarial
        if USE_TENSORBOARD:
            training_history = adversarial_model.fit(x_training, y_training, epochs=number_epochs, batch_size=batch_size,
                                                     verbose=DEFAULT_VERBOSE_LIST[verbose_level], callbacks=callbacks)
        else:
            training_history = adversarial_model.fit(x_training, y_training, epochs=number_epochs, batch_size=batch_size,
                                                     verbose=DEFAULT_VERBOSE_LIST[verbose_level])

        logging.info("Treinamento concluído.")

        ## Salvar os modelos treinados, se solicitado
        if save_models:
            adversarial_model.save_models(output_dir, i)

        ## Plotar a curva de perda durante o treinamento
        generator_loss_list = training_history.history['loss_g']
        discriminator_loss_list = training_history.history['loss_d']
        plot_loss_curve_instance = PlotCurveLoss()
        plot_loss_curve_instance.plot_training_loss_curve(generator_loss_list, discriminator_loss_list, output_dir, i,
                                                          path_curve_loss)

        ## Preparação para a geração de dados sintéticos
        number_samples_true = len([positional_label for positional_label in y_test.tolist() if positional_label == 1])
        number_samples_false = len([positional_label for positional_label in y_test.tolist() if positional_label == 0])
        num_samples_true_desired = len([positional_label for positional_label in y_training.tolist() if positional_label == 1])
        num_samples_false_desired = len([positional_label for positional_label in y_training.tolist() if positional_label == 0])

        ## Geração de amostras sintéticas para as classes verdadeira e falsa
        x_true_synthetic, y_true_synthetic = generate_samples(adversarial_model, num_samples_true_desired, latent_dim,
                                                              1, verbose_level, latent_mean_distribution,
                                                              latent_stander_deviation)
        x_false_synthetic, y_false_synthetic = generate_samples(adversarial_model, num_samples_false_desired,
                                                                latent_dim, 0, verbose_level, latent_mean_distribution,
                                                                latent_stander_deviation)
        ## Juntar os dados sintéticos gerados para ambas as classes
        x_synthetic_samples = np.concatenate((x_true_synthetic, x_false_synthetic), axis=0)
        y_synthetic_samples = np.rint(np.concatenate((y_true_synthetic, y_false_synthetic), axis=0))
        y_synthetic_samples = np.squeeze(y_synthetic_samples, axis=1)

        ## Converter para DataFrame e adicionar os nomes das colunas
        synthetic_columns = dataset.columns[:-1]
        df_synthetic = pd.DataFrame(data=x_synthetic_samples, columns=synthetic_columns)
        df_synthetic['class'] = y_synthetic_samples

            ## Salvar dados sintéticos em um arquivo CSV
        synthetic_filename = f'synthetic_data_fold_{i + 1}.csv'
        synthetic_filepath = os.path.join(output_dir, synthetic_filename)
        df_synthetic.to_csv(synthetic_filepath, index=False, sep=',', header=True)

        ## Realizar o transform sobre os dados de avaliação  para gerar o segundo conjunto sintético
        x_synthetic_training=x_synthetic_samples
        y_synthetic_training=y_synthetic_samples
        print(x_synthetic_training.shape)
        print(y_synthetic_training.shape)
        if USE_TENSORBOARD:
          adversarial_model.fit(x_test, y_test)
        else:
          adversarial_model.fit(x_test, y_test)

        ## Gerar amostras sintéticas com o número desejado para cada classe
        x_true_synthetic, y_true_synthetic = generate_samples(adversarial_model, number_samples_true, latent_dim,
                                                              1, verbose_level, latent_mean_distribution,
                                                              latent_stander_deviation)
        x_false_synthetic, y_false_synthetic = generate_samples(adversarial_model, number_samples_false, latent_dim,
                                                                0, verbose_level, latent_mean_distribution,
                                                                latent_stander_deviation)
        x_synthetic_samples = np.concatenate((x_true_synthetic, x_false_synthetic), dtype=dataset_type)
        y_synthetic_samples = np.rint(np.concatenate((y_true_synthetic, y_false_synthetic)))
        y_synthetic_samples = np.squeeze(y_synthetic_samples, axis=1)

        ## Obter os classificadores treinados com dados reais (TR_As) e sintéticos (TS_Ar)
        list_classifiers_TR_As = instance_classifier_TR_As.get_trained_classifiers(classifier_list, x_training, y_training,
                                                                                dataset_type, verbose_level, input_data_shape)
        list_classifiers_TS_Ar = instance_classifier_TS_Ar.get_trained_classifiers(classifier_list, x_synthetic_training, y_synthetic_training,
                                                                                dataset_type, verbose_level, input_data_shape)

        ## Avaliar oa performance de ambos classificadores e coletar as métricas de utilidade
        evaluate_TR_As_data(list_classifiers_TR_As, x_synthetic_samples, y_synthetic_samples, i, k, True,
                           output_dir, classifier_list, title_output, path_confusion_matrix, verbose_level, dict_metrics, dict_TR_As_auc)
        evaluate_TS_Ar_data(list_classifiers_TS_Ar, x_test, y_test, i, k, True, output_dir, classifier_list,
                           title_output, path_confusion_matrix, verbose_level, dict_metrics, dict_TS_Ar_auc)

        ## Calcular métricas de fidelidade entre dados sintéticos e reais
        uni, counts = np.unique(y_test, return_counts=True)
        indexation = min(counts[0], counts[1])
        falses = x_test[y_test == 0][:indexation]
        positives = x_test[y_test == 1][:indexation]
        x_false_synthetic = x_false_synthetic[:indexation]
        x_true_synthetic = x_true_synthetic[:indexation]
        comparative_metrics1 = comparative_data(i, x_false_synthetic, falses, "falses")
        comparative_metrics2 = comparative_data(i, x_true_synthetic, positives, "true")

        ## Armazenar as métricas de fidelidade entre os dados
        dict_similarity["list_mean_squared_error"]["false"].append(comparative_metrics1[0])
        dict_similarity["list_cosine_similarity"]["false"].append(comparative_metrics1[1])
        dict_similarity["list_maximum_mean_discrepancy"]["false"].append(comparative_metrics1[2])
        dict_similarity["list_mean_squared_error"]["positive"].append(comparative_metrics2[0])
        dict_similarity["list_cosine_similarity"]["positive"].append(comparative_metrics2[1])
        dict_similarity["list_maximum_mean_discrepancy"]["positive"].append(comparative_metrics2[2])

    ## Exibir e exportar os resultados finais
    show_and_export_results(dict_similarity,classifier_list, output_dir, title_output,dict_metrics,dict_TR_As_auc,dict_TS_Ar_auc)

def show_all_settings(arg_parsers):
        """
        Função exibe todas as configurações do experimento nos logs.

        Parâmetros:
            - arg_parsers: Objeto contendo os argumentos do experimento.
        """

        ## Exibe o comando de execução no log
        logging.info("Command:\n\t{0}\n".format(" ".join([x for x in sys.argv])))

        ## Exibe as configurações no log
        logging.info("Settings:")

        ## Calcula o comprimento máximo das chaves dos argumentos para formatação
        lengths = [len(x) for x in vars(arg_parsers).keys()]
        max_length = max(lengths)

        ## Itera sobre os argumentos, formata e exibe cada um no log
        for k, v in sorted(vars(arg_parsers).items()):
            settings_parser = "\t"
            settings_parser += k.ljust(max_length, " ")
            settings_parser += " : {}".format(v)
            logging.info(settings_parser)

        logging.info("")
    

def initial_step(initial_arguments, dataset_type):
    """
    A função realiza:
    - A configuração inicial do experimento,
    - Salvamento dos argumentos da linha de comando,
    - Carregamento do dataset e estabelecimento da forma dos dados de entrada.


    Parâmetros:
        - initial_arguments: Objeto contendo os argumentos iniciais do experimento.
        - dataset_type: Tipo de dados do dataset.

    Retorna:
        - dataset_file_loaded: DataFrame do dataset carregado e pré-processado.
        - dataset_input_shape: Forma dos dados de entrada.
        - dataset_labels: String contendo o nome do arquivo do dataset carregado.
    """

    ## Caminho para o arquivo onde os argumentos da linha de comando serão salvos
    file_args = os.path.join(initial_arguments.output_dir, 'commandline_args.txt')

    ## Salva os argumentos da linha de comando em um arquivo JSON
    with open(file_args, 'w') as f:
        json.dump(initial_arguments.__dict__, f, indent=2)

    ## Carrega o dataset a partir do arquivo CSV e remove valores ausentes (NaN)
    dataset_file_loaded = pd.read_csv(initial_arguments.input_dataset, dtype=dataset_type)
    dataset_file_loaded = dataset_file_loaded.dropna()

    ## Calcula a forma dos dados de entrada (número de colunas menos a coluna de rótulo)
    dataset_input_shape = dataset_file_loaded.shape[1] - 1

    ## Obtém o nome do arquivo do dataset
    input_dataset = os.path.basename(initial_arguments.input_dataset)
    dataset_labels = f'Dataset: {input_dataset}'

    return dataset_file_loaded, dataset_input_shape, dataset_labels
def create_argparse():
    """
    Estabelece a lista de parâmetros de entradas para o argparse. 
    """
    parser = argparse.ArgumentParser(description='Run the experiment with cGAN and classifiers')

    parser.add_argument('-i', '--input_dataset', type=str, required=True,
                        help='Arquivo do dataset de entrada (Formato CSV)')

    parser.add_argument('-c', '--classifier', type=list_of_strs, default=DEFAULT_CLASSIFIER_LIST,
                        help='Classificador (ou lista de classificadores separada por ,) padrão:{}.'.format(
                            DEFAULT_CLASSIFIER_LIST))

    parser.add_argument('-o', '--output_dir', type=str,
                        default=f'out_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
                        help='Diretório para gravação dos arquivos de saída.')

    parser.add_argument('--data_type', type=str, default=DEFAULT_DATA_TYPE,
                        choices=['int8', 'float16', 'float32'],
                        help='Tipo de dado para representar as características das amostras.')

    parser.add_argument('--num_samples_class_malware', type=int,required=True,
                        default=DEFAULT_NUMBER_GENERATE_MALWARE_SAMPLES,
                        help='Número de amostras da Classe 1 (maligno).')

    parser.add_argument('--num_samples_class_benign', type=int,required=True,
                        default=DEFAULT_NUMBER_GENERATE_BENIGN_SAMPLES,
                        help='Número de amostras da Classe 0 (benigno).')

    parser.add_argument('--number_epochs', type=int,
                        default=DEFAULT_NUMBER_EPOCHS_CONDITIONAL_GAN,
                        help='Número de épocas (iterações de treinamento).')

    parser.add_argument('--k_fold', type=int,
                        default=DEFAULT_NUMBER_STRATIFICATION_FOLD,
                        help='Número de folds para validação cruzada.')

    parser.add_argument('--initializer_mean', type=float,
                        default=DEFAULT_ADVERSARIAL_INITIALIZER_MEAN,
                        help='Valor central da distribuição gaussiana do inicializador.')

    parser.add_argument('--initializer_deviation', type=float,
                        default=DEFAULT_ADVERSARIAL_INITIALIZER_DEVIATION,
                        help='Desvio padrão da distribuição gaussiana do inicializador.')

    parser.add_argument("--latent_dimension", type=int,
                        default=DEFAULT_ADVERSARIAL_LATENT_DIMENSION,
                        help="Dimensão do espaço latente para treinamento cGAN")


    parser.add_argument("--training_algorithm", type=str,
                        default=DEFAULT_ADVERSARIAL_TRAINING_ALGORITHM,
                        help="Algoritmo de treinamento para cGAN.",
                        choices=['Adam'])

    parser.add_argument("--activation_function",
                        type=str, default=DEFAULT_ADVERSARIAL_ACTIVATION,
                        help="Função de ativação da cGAN.",
                        choices=['LeakyReLU', 'ReLU', 'PReLU'])

    parser.add_argument("--dropout_decay_rate_g",
                        type=float, default=DEFAULT_ADVERSARIAL_DROPOUT_DECAY_RATE_G,
                        help="Taxa de decaimento do dropout do gerador da cGAN")

    parser.add_argument("--dropout_decay_rate_d",
                        type=float, default=DEFAULT_ADVERSARIAL_DROPOUT_DECAY_RATE_D,
                        help="Taxa de decaimento do dropout do discriminador da cGAN")

    parser.add_argument("--dense_layer_sizes_g", type=list_of_ints, nargs='+',
                        default=DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_G,
                        help=" Valor das camadas densas do gerador")

    parser.add_argument("--dense_layer_sizes_d", type=list_of_ints, nargs='+',
                        default=DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_D,
                        help="valor das camadas densas do discriminador")

    parser.add_argument('--batch_size', type=int,
                        default=DEFAULT_ADVERSARIAL_BATCH_SIZE,
                        choices=[16, 32, 64,128,256],
                        help='Tamanho do lote da cGAN.')

    parser.add_argument("--verbosity", type=int,
                        help='Verbosity (Default {})'.format(DEFAULT_VERBOSITY),
                        default=DEFAULT_VERBOSITY)

    parser.add_argument("--save_models", type=bool,
                        help='Salvar modelos treinados (Default {})'.format(DEFAULT_SAVE_MODELS),
                        default=DEFAULT_SAVE_MODELS)

    parser.add_argument("--path_confusion_matrix", type=str,
                        help='Diretório de saída das matrizes de confusão',
                        default=DEFAULT_OUTPUT_PATH_CONFUSION_MATRIX)

    parser.add_argument("--path_curve_loss", type=str,
                        help='Diretório de saída dos gráficos de curva de treinamento',
                        default=DEFAULT_OUTPUT_PATH_TRAINING_CURVE)

    parser.add_argument("--latent_mean_distribution", type=float,
                        help='Média da distribuição do ruído aleatório de entrada',
                        default=DEFAULT_ADVERSARIAL_RANDOM_LATENT_MEAN_DISTRIBUTION)

    parser.add_argument("--latent_stander_deviation", type=float,
                        help='Desvio padrão do ruído aleatório de entrada',
                        default=DEFAULT_ADVERSARIAL_RANDOM_LATENT_STANDER_DEVIATION)
    parser.add_argument('-a','--use_aim',action='store_true',help="Uso ou não da ferramenta aim para monitoramento") 
    
    parser.add_argument('-ml','--use_mlflow',action='store_true',help="Uso ou não da ferramenta mlflow para monitoramento") 

    parser.add_argument('-rid','--run_id',type=str,help="codigo da run",default=None) 
    
    
    parser.add_argument("-tb",'--use_tensorboard',action='store_true',help="Uso ou não da ferramenta tensorboard para monitoramento")

    return parser.parse_args()



if __name__ == "__main__":
    """
    Função principal responsável por chamar as funções necessarias para a configuração e execução do código
    """

    arguments = create_argparse()

    logging_format = '%(asctime)s\t***\t%(message)s'

    ## Configura o mecanismo de logging
    if arguments.verbosity == logging.DEBUG:
        ## mostra mais detalhes
        logging_format = '%(asctime)s\t***\t%(levelname)s {%(module)s} [%(funcName)s] %(message)s'

    ##Verifica se o caminho para o diretório de output existe
    Path(arguments.output_dir).mkdir(parents=True, exist_ok=True)
    logging_filename = os.path.join(arguments.output_dir, LOGGING_FILE_NAME)


    logging.basicConfig(format=logging_format, level=arguments.verbosity)

    ## Adiciona o arquivo de log com os valores de logging estabelecidos nos parâmetros de entrada
    rotatingFileHandler = RotatingFileHandler(filename=logging_filename, maxBytes=100000, backupCount=5)
    rotatingFileHandler.setLevel(arguments.verbosity)
    rotatingFileHandler.setFormatter(logging.Formatter(logging_format))
    logging.getLogger().addHandler(rotatingFileHandler)
    
    show_all_settings(arguments)
    
    ##Realiza a cronometrização do tempo de de início da execução
    time_start_campaign = datetime.datetime.now()
    ## Leitura dos argumentos de entrada dos parâmetros da ferramenta
    if arguments.data_type == 'int8':
        data_type = np.int8

    elif arguments.data_type == 'float16':
        data_type = np.float16

    else:
        data_type = np.float32

    if arguments.dense_layer_sizes_g != DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_G:
        arguments.dense_layer_sizes_g = arguments.dense_layer_sizes_g[0]

    if arguments.dense_layer_sizes_d != DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_D:
        arguments.dense_layer_sizes_d = arguments.dense_layer_sizes_d[0]

    if arguments.classifier != DEFAULT_CLASSIFIER_LIST:
        arguments.classifier = arguments.classifier[0]
    if arguments.use_aim == True:
        USE_AIM= True
    if arguments.use_tensorboard:
        USE_TENSORBOARD=True
    if arguments.use_mlflow:
         USE_MLFLOW= True
    ## Caso o uso da ferramento aimstack esteja sendo utilizado é necessário estabelecer o diretório e o nome do experimento
    if USE_AIM:
        output_dir = arguments.output_dir
        experiment_name= output_dir.split('/')[-1]
        aim_run=Run(experiment=experiment_name)
    if USE_MLFLOW:
       ##Estabelece o endereço do servidor de rastreamento mlflow como localhost porta 6002
       mlflow.set_tracking_uri("http://127.0.0.1:6002/")
       
       if arguments.run_id==None:
             ## Resumir uma execução caso tenha sido abortada
             mlflow.start_run()
             mlflow.set_experiment(arguments.output_dir)
       else:
             ##inicializa a nova execução
             mlflow.start_run(run_id=arguments.run_id)
    if USE_TENSORBOARD:
      ##estabelece os parâmetros de saida da ferramenta TensorBoard
      log_folder = "tensorboardfolder/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
      callbacks=[TensorBoard(log_dir=log_folder,histogram_freq=1,write_graph=True,write_images=True,update_freq='epoch',profile_batch=2,embeddings_freq=1)]
      file_writer = tf.summary.create_file_writer(log_folder)

    dataset_file, output_shape, output_label = initial_step(arguments, data_type)
    run_experiment(dataset_file, output_shape,
                   arguments.k_fold,
                   arguments.classifier, arguments.output_dir, batch_size=arguments.batch_size,
                   training_algorithm=arguments.training_algorithm, number_epochs=arguments.number_epochs,
                   latent_dim=arguments.latent_dimension, activation_function=arguments.activation_function,
                   dropout_decay_rate_g=arguments.dropout_decay_rate_g,
                   dropout_decay_rate_d=arguments.dropout_decay_rate_d,
                   dense_layer_sizes_g=arguments.dense_layer_sizes_g, dense_layer_sizes_d=arguments.dense_layer_sizes_d,
                   dataset_type=data_type, title_output=output_label, initializer_mean=arguments.initializer_mean,
                   initializer_deviation=arguments.initializer_deviation, save_models=arguments.save_models,
                   path_confusion_matrix=arguments.path_confusion_matrix, path_curve_loss=arguments.path_curve_loss,
                   verbose_level=arguments.verbosity, latent_mean_distribution=arguments.latent_mean_distribution,
                   latent_stander_deviation=arguments.latent_stander_deviation, num_samples_class_malware=arguments.num_samples_class_malware, num_samples_class_benign=arguments.num_samples_class_benign )
    ##realiza a cronometrização do tempo de fim da execução
    time_end_campaign = datetime.datetime.now()
    if USE_TENSORBOARD:
       file_writer.close()