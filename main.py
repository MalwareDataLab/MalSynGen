# -*- coding: utf-8 -*-
__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2022/06/01'
__last_update__ = '2023/08/03'
__credits__ = ['unknown']


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
    from Tools.tools import PlotRegressiveMetrics
    from Tools.tools import PlotCurveLoss
    from Tools.tools import ProbabilisticMetrics
    from Tools.tools import PlotClassificationMetrics
    from Tools.tools import P_values_metrics
    from Tools.tools import DEFAULT_COLOR_NAME 
    from sklearn.linear_model import SGDClassifier
    import neptune
    from mlflow.models import infer_signature
    from tensorflow.keras.callbacks import TensorBoard
    import io
    import pickle   
    USE_AIM=False
    USE_MLFLOW=False
    USE_NEPTUNE=False
    USE_TENSORBOARD=False


    os.environ["KERAS_BACKEND"] = "tensorflow"
except ImportError as error:

    print(error)
    print()
    print("1. (optional) Setup a virtual environment: ")
    print("  python3 - m venv ~/Python3env/SynTabData ")
    print("  source ~/Python3env/SynTabData/bin/activate ")
    print()
    print("2. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*the default value of `keepdims` will become False.*")
    warnings.filterwarnings("ignore", message="Variables are collinear")

DEFAULT_VERBOSITY = logging.INFO
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'
DEFAULT_DATA_TYPE = "float32"

DEFAULT_NUMBER_GENERATE_MALWARE_SAMPLES = 2000
DEFAULT_NUMBER_GENERATE_BENIGN_SAMPLES = 2000
DEFAULT_NUMBER_EPOCHS_CONDITIONAL_GAN = 100
DEFAULT_NUMBER_STRATIFICATION_FOLD = 5

DEFAULT_ADVERSARIAL_LATENT_DIMENSION = 128
DEFAULT_ADVERSARIAL_TRAINING_ALGORITHM = "Adam"
DEFAULT_ADVERSARIAL_ACTIVATION = "LeakyReLU"  # ['LeakyReLU', 'ReLU', 'PReLU']
DEFAULT_ADVERSARIAL_DROPOUT_DECAY_RATE_G = 0.2
DEFAULT_ADVERSARIAL_DROPOUT_DECAY_RATE_D = 0.4
DEFAULT_ADVERSARIAL_INITIALIZER_MEAN = 0.0
DEFAULT_ADVERSARIAL_INITIALIZER_DEVIATION = 0.02
DEFAULT_ADVERSARIAL_BATCH_SIZE = 32
DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_G = [512]
DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_D = [512]
DEFAULT_ADVERSARIAL_RANDOM_LATENT_MEAN_DISTRIBUTION = 0.0
DEFAULT_ADVERSARIAL_RANDOM_LATENT_STANDER_DEVIATION = 1.0

DEFAULT_CONDITIONAL_LAST_ACTIVATION_LAYER = "sigmoid"
DEFAULT_PERCEPTRON_TRAINING_ALGORITHM = "Adam"
DEFAULT_PERCEPTRON_LOSS = "binary_crossentropy"
DEFAULT_PERCEPTRON_DENSE_LAYERS_SETTINGS = [512, 256, 256]
DEFAULT_PERCEPTRON_DROPOUT_DECAY_RATE = 0.2
DEFAULT_PERCEPTRON_METRIC = ["accuracy"]
DEFAULT_SAVE_MODELS = True
DEFAULT_OUTPUT_PATH_CONFUSION_MATRIX = "confusion_matrix"
DEFAULT_OUTPUT_PATH_TRAINING_CURVE = "training_curve"
DEFAULT_CLASSIFIER_LIST = ["RandomForest", "DecisionTree", "AdaBoost","Perceptron","SVM","SGDRegressor","XGboost"]  #"AdaBoost", "SupportVectorMachine", "QuadraticDiscriminant", "NaiveBayes","Perceptron"
#"AdaBoost"
DEFAULT_VERBOSE_LIST = {logging.INFO: 2, logging.DEBUG: 1, logging.WARNING: 2,
                        logging.FATAL: 0, logging.ERROR: 0}

LOGGING_FILE_NAME = "logging.log"
aim_run=None
nept_run=None
callbacks=None
file_writer=None

# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))


# Define a custom argument type for a list of integers
def list_of_strs(arg):
    return list(map(str, arg.split(',')))
def generate_samples(instance_model, number_instances, latent_dimension, label_class, verbose_level,
                     latent_mean_distribution, latent_stander_deviation):

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
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    digit = tf.image.decode_png(buf.getvalue(), channels=4)
    digit = tf.expand_dims(digit, 0)

    return digit

def comparative_data(fold, x_synthetic, real_data,label):
    instance_metrics = ProbabilisticMetrics()
    synthetic_mean_squared_error = instance_metrics.get_mean_squared_error(real_data, x_synthetic)
    synthetic_cosine_similarity = instance_metrics.get_cosine_similarity(real_data, x_synthetic)
    #synthetic_kl_divergence = instance_metrics.get_kl_divergence(real_data, x_synthetic
    synthetic_maximum_mean_discrepancy = instance_metrics.get_maximum_mean_discrepancy(real_data, x_synthetic)
    logging.info(f"Similarity Metrics")
    logging.info(f"  Synthetic Fold {fold + 1} {label} - Mean Squared Error: " + str(synthetic_mean_squared_error))
    logging.info(f"  Synthetic Fold {fold + 1} {label} - Cosine Similarity: " + str(synthetic_cosine_similarity))
    #logging.info(f"  Synthetic Fold {fold + 1} - KL Divergence: " + str(synthetic_kl_divergence))
    logging.info(f"  Synthetic Fold {fold + 1} {label} - Maximum Mean Discrepancy: " + str(synthetic_maximum_mean_discrepancy))
    logging.info("")

    return [synthetic_mean_squared_error,synthetic_cosine_similarity,synthetic_maximum_mean_discrepancy]


def evaluate_TRTS_data(list_classifiers, x_TRTS, y_TRTS, fold, k, generate_confusion_matrix,
                            output_dir, classifier_type, out_label, path_confusion_matrix, verbose_level,dict,syn_aucs,dict_log_likehood_syn):
    instance_metrics = ProbabilisticMetrics()
    accuracy_TRTS_list, precision_TRTS_list, recall_TRTS_list, f1_score_TRTS_list = [], [], [], []
    y_predict_prob=[]
    logging.info(f"TRTS Fold {fold + 1}/{k} results\n")
  
    for index, classifier_model in enumerate(list_classifiers):

        if classifier_type[index] == "Perceptron":
            y_predicted_TRTS = classifier_model.predict(x_TRTS, verbose=DEFAULT_VERBOSE_LIST[verbose_level])
            y_predicted_TRTS = np.rint(np.squeeze(y_predicted_TRTS, axis=1))
            y_predict_prob=y_predicted_TRTS
           
            #y_predict_prob=y_predicted_TRTS.argmax(axis=-1)
           # y_predict_prob=np.argmax(y_predict_prob,axis=1, keepdims=True)
             
          #  y_predicted_prob = np.rint(np.squeeze(y_predicted_prob, axis=1))
           # print(y_predicted_prob,"prob ")
           # print(len(y_predicted_prob),"len")
        else:

            y_predicted_TRTS = classifier_model.predict(x_TRTS)
            y_predict_prob=classifier_model.predict_proba(x_TRTS)[::,1]
            #y_predict_prob=np.argmax(y_predict_prob,axis=1, keepdims=True)
        y_predicted_TRTS = y_predicted_TRTS.astype(int)
        y_TRTS = y_TRTS.astype(int)

        confusion_matrix_TRTS = confusion_matrix(y_TRTS, y_predicted_TRTS)
        accuracy_TRTS = instance_metrics.get_accuracy(y_TRTS, y_predicted_TRTS)
        precision_TRTS = instance_metrics.get_precision(y_TRTS, y_predicted_TRTS)
        recall_TRTS = instance_metrics.get_recall(y_TRTS, y_predicted_TRTS)
        f1_score_TRTS = instance_metrics.get_f1_score(y_TRTS, y_predicted_TRTS)
        logging.info(f" Classifier Model: {classifier_type[index]}")
        logging.info(f"   TRTS Fold {fold + 1} - Confusion Matrix:")
        logging.info(confusion_matrix_TRTS)
        logging.info(f"\n   Classifier Metrics:")
        logging.info(f"     TRTS Fold {fold + 1} - Accuracy: " + str(accuracy_TRTS))
        logging.info(f"     TRTS Fold {fold + 1} - Precision: " + str(precision_TRTS))
        logging.info(f"     TRTS Fold {fold + 1} - Recall: " + str(recall_TRTS))
        logging.info(f"     TRTS Fold {fold + 1} - F1 Score: " + str(f1_score_TRTS) + "\n")
        ac='TRTS Accuracy '+f' Classifier Model {classifier_type[index]}'
        pc='TRTS Precision '+f' Classifier Model {classifier_type[index]}'
        rr='TRTS Recall '+f' Classifier Model {classifier_type[index]}'
        f1='TRTS f1 Score '+f' Classifier Model {classifier_type[index]}'

        values={ac:accuracy_TRTS,pc:precision_TRTS,rr:recall_TRTS,f1:f1_score_TRTS}
        dict["TRTS accuracy"][classifier_type[index]].append(accuracy_TRTS)
        dict["TRTS precision"][classifier_type[index]].append(precision_TRTS)
        dict["TRTS F1 score"][classifier_type[index]].append(f1_score_TRTS)
        dict["TRTS recall"][classifier_type[index]].append(recall_TRTS)
        fpr, tpr,thresholds=sklearn.metrics.roc_curve(y_TRTS,   y_predict_prob)
        log_likehood=log_loss(y_TRTS,y_predict_prob)
        dict_log_likehood_syn[classifier_type[index]].append(log_likehood)
        plt.figure()  
        Path(os.path.join(output_dir, path_confusion_matrix)).mkdir(parents=True, exist_ok=True)
        roc_file = os.path.join(output_dir, path_confusion_matrix,f'Roc_curve_TRTS_{classifier_type[index]}_k{fold + 1}.jpg')
        auc=metrics.roc_auc_score(y_TRTS,   y_predict_prob)
        plt.plot(fpr, tpr)
        plt.savefig(roc_file, bbox_inches="tight")
        syn_aucs[classifier_type[index]].append(auc)

        if generate_confusion_matrix:
            plt.figure()
            selected_color_map = plt.colormaps.get_cmap(DEFAULT_COLOR_NAME[(fold + 2) % len(DEFAULT_COLOR_NAME)])
            confusion_matrix_instance = PlotConfusionMatrix()
            confusion_matrix_instance.plot_confusion_matrix(confusion_matrix_TRTS, out_label, selected_color_map)
            Path(os.path.join(output_dir, path_confusion_matrix)).mkdir(parents=True, exist_ok=True)
            matrix_file = os.path.join(output_dir, path_confusion_matrix,
                                       f'CM_TRTS_{classifier_type[index]}_k{fold + 1}.jpg')
            plt.savefig(matrix_file, bbox_inches='tight')
            if USE_AIM:
               aim_run.track(values)
               plt.savefig(matrix_file, bbox_inches='tight')
               aim_image = Image(matrix_file)
               aim_run.track(value=aim_image,name= f'CM_TRTS_{classifier_type[index]}_k{fold + 1}')
            if USE_MLFLOW:
                  mlflow.log_metrics(values,step=fold+1)
                  mlflow.log_artifact(matrix_file, 'images')
            if USE_TENSORBOARD:
               with file_writer.as_default():
                  cm_image = plot_to_image(matrix_file)
                  tf.summary.image(f'CM_TSTR_{classifier_type[index]}_k{fold + 1}', cm_image,step=fold+1)
                  tf.summary.scalar(ac, data=accuracy_TRTS, step=fold+1)
                  tf.summary.scalar(pc, data=precision_TRTS, step=fold+1)
                  tf.summary.scalar(rr, data=recall_TRTS, step=fold+1)
                  tf.summary.scalar(f1, data=f1_score_TRTS, step=fold+1)
        accuracy_TRTS_list.append(accuracy_TRTS)
        precision_TRTS_list.append(precision_TRTS)
        recall_TRTS_list.append(recall_TRTS)
        f1_score_TRTS_list.append(f1_score_TRTS)

    return [accuracy_TRTS_list, precision_TRTS_list, recall_TRTS_list, f1_score_TRTS_list]


def evaluate_TSTR_data(list_classifiers, x_TSTR, y_TSTR, fold, k, generate_confusion_matrix, output_dir,
                       classifier_type, out_label, path_confusion_matrix, verbose_level,dict,TSTR_aucs,dict_log_likehood_TSTR):
    logging.info(f"TSTR Fold {fold + 1}/{k} results")
    instance_metrics = ProbabilisticMetrics()
    y_predict_prob=[]
    accuracy_TSTR_list, precision_TSTR_list, recall_TSTR_list, f1_TSTR_list = [], [], [], []
    for index, classifier_model in enumerate(list_classifiers):

        if classifier_type[index] == "Perceptron":
            y_predicted_TSTR = classifier_model.predict(x_TSTR, verbose=DEFAULT_VERBOSE_LIST[verbose_level])
            y_predicted_TSTR = np.rint(np.squeeze(y_predicted_TSTR, axis=1))
            y_predict_prob=y_predicted_TSTR
            #y_predict_prob=y_predicted_TSTR.argmax(axis=-1)
            #calibrated_clf = CalibratedClassifierCV(classifier_model, cv="prefit")
            #calibrated_clf.fit(X_calib, y_calib)
           # y_predict_prob=np.argmax(y_predict_prob,axis=1)


        else:
            print(classifier_type[index])
            y_predicted_TSTR = classifier_model.predict(x_TSTR)
            y_predict_prob=classifier_model.predict_proba(x_TSTR)[::,1]
            #y_predict_prob=np.argmax(y_predict_prob,axis=1)

        y_predicted_TSTR = y_predicted_TSTR.astype(int)
        
        y_sample_TSTR = y_TSTR.astype(int)
        confusion_matrix_TSTR = confusion_matrix(y_sample_TSTR, y_predicted_TSTR)
        accuracy_TSTR = instance_metrics.get_accuracy(y_sample_TSTR, y_predicted_TSTR)
        precision_TSTR = instance_metrics.get_precision(y_sample_TSTR, y_predicted_TSTR)
        recall_TSTR = instance_metrics.get_recall(y_sample_TSTR, y_predicted_TSTR)
        f1_TSTR = instance_metrics.get_f1_score(y_sample_TSTR, y_predicted_TSTR)
 
        logging.info(f" Classifier Model: {classifier_type[index]}")
        logging.info(f"   TSTR Fold {fold + 1} - Confusion Matrix:")
        logging.info(confusion_matrix_TSTR)
        logging.info(f"\n   Classifier Metrics:")
        logging.info(f"     TSTR Fold {fold + 1} - Accuracy: " + str(accuracy_TSTR))
        logging.info(f"     TSTR Fold {fold + 1} - Precision: " + str(precision_TSTR))
        logging.info(f"     TSTR Fold {fold + 1} - Recall: " + str(recall_TSTR))
        logging.info(f"     TSTR Fold {fold + 1} - F1 Score: " + str(f1_TSTR) + "\n")
        logging.info("")
        ac='TSTR Accuracy '+f' Classifier Model {classifier_type[index]}'
        pc='TSTR precision '+f' Classifier Model {classifier_type[index]}'
        rr='TSTR Recall '+f' Classifier Model {classifier_type[index]}'
        f1='TSTR f1 Score '+f' Classifier Model {classifier_type[index]}'
        values={ac:accuracy_TSTR,pc:precision_TSTR,rr:recall_TSTR,f1:f1_TSTR}

        dict["TSTR accuracy"][classifier_type[index]].append(accuracy_TSTR)
        dict["TSTR precision"][classifier_type[index]].append(precision_TSTR)
        dict["TSTR F1 score"][classifier_type[index]].append(f1_TSTR)
        dict["TSTR recall"][classifier_type[index]].append(recall_TSTR)

        fpr, tpr,thresholds=metrics.roc_curve(y_sample_TSTR, y_predict_prob)
        log_likehood=log_loss(y_sample_TSTR,y_predict_prob)
        dict_log_likehood_TSTR[classifier_type[index]].append(log_likehood)
        plt.figure()  
        Path(os.path.join(output_dir, path_confusion_matrix)).mkdir(parents=True, exist_ok=True)
        roc_file = os.path.join(output_dir, path_confusion_matrix,f'Roc_curve_TSTR_{classifier_type[index]}_k{fold + 1}.jpg')
        auc=metrics.roc_auc_score(y_sample_TSTR,  y_predict_prob)
        plt.plot(fpr, tpr)
        plt.savefig(roc_file, bbox_inches="tight")
        TSTR_aucs[classifier_type[index]].append(auc)
        if generate_confusion_matrix:
            plt.figure()
            selected_color_map = plt.colormaps.get_cmap(DEFAULT_COLOR_NAME[(fold + 2) % len(DEFAULT_COLOR_NAME)])
            confusion_matrix_instance = PlotConfusionMatrix()
            confusion_matrix_instance.plot_confusion_matrix(confusion_matrix_TSTR, out_label, selected_color_map)
            Path(os.path.join(output_dir, path_confusion_matrix)).mkdir(parents=True, exist_ok=True)
            matrix_file = os.path.join(output_dir, path_confusion_matrix,
                                       f'CM_TSTR_{classifier_type[index]}_k{fold + 1}.jpg')
            matrix=plt.savefig(matrix_file, bbox_inches='tight')
            cm_image = plot_to_image(matrix_file)

            if USE_TENSORBOARD:
               with file_writer.as_default():
                  tf.summary.image(f'CM_TSTR_{classifier_type[index]}_k{fold + 1}', cm_image,step=fold+1)
                  tf.summary.scalar(ac, data=accuracy_TSTR, step=fold+1)
                  tf.summary.scalar(pc, data=precision_TSTR, step=fold+1)
                  tf.summary.scalar(rr, data=recall_TSTR, step=fold+1)
                  tf.summary.scalar(f1, data=f1_TSTR, step=fold+1)

            if USE_AIM:
               aim_run.track(values)
               plt.savefig(matrix_file, bbox_inches='tight')
               aim_image = Image(matrix_file)
               aim_run.track(value=aim_image, name=f'CM_TSTR_{classifier_type[index]}_k{fold + 1}')
     #                 context={'classifier_type': classifier_type[index]})
            if USE_MLFLOW:
                  mlflow.log_metrics(values,step=fold+1)
                  mlflow.log_artifact(matrix_file, 'images')

        accuracy_TSTR_list.append(accuracy_TSTR)
        precision_TSTR_list.append(precision_TSTR)
        recall_TSTR_list.append(recall_TSTR)
        f1_TSTR_list.append(f1_TSTR)


    return [accuracy_TSTR_list, precision_TSTR_list, recall_TSTR_list, f1_TSTR_list]

def p_value_test (Real_label,false_label,type_of_metric,classifier_type):
    real=[]
    synt=[]
    estaticts=0
    p_value=0
    aux1=[]
    aux2=[]
    #print(classifier_type)
    #for index in range(len(classifier_type)):
    aux1=np.asfarray(Real_label[classifier_type])
    aux2=np.asfarray(false_label[classifier_type])   
    estaticts, p_value = stats.wilcoxon(aux1,aux2,alternative='two-sided')
    logging.info("  P_value of {}  {} and stats {} of {} \n".format(type_of_metric,p_value,estaticts,classifier_type))
    return p_value
    #plot_filename = os.path.join(output_dir, f'{classifier_type[index]}_p_values.pdf')
    
def wilcoxon_test_per_fold(list_syn,list_real):
    aux_real=[]
    aux_false=[]
    estaticts=0
    p_value=0
    for i,j in zip(list_syn,list_real) :
      aux_real.append(i)
      aux_false.append(j)
    estaticts, p_value = stats.wilcoxon(aux_real,aux_false,zero_method="zsplit")
    return p_value
def show_and_export_results(dict_similarity,classifier_type,output_dir,title_output_label,dict_metrics,dict_TRTS_auc,dict_TSTR_auc,dict_log_likehood_TSTR,dict_log_likehood_TRTS):
    plot_classifier_metrics = PlotClassificationMetrics()
    plot_regressive_metrics = PlotRegressiveMetrics()
    p_metrics=P_values_metrics()
    #dict_list=[dict_metrics,dict_metrics]
    dict_to_csv={}
    analysis_type_list=["Treinado com TSTR","Treinado com sint"]
    
    for index in range(len(classifier_type)):
        #for analysis_type in zip(analysis_type_list):
        logging.info("Overall TRTS Results: Classifier {}\n".format(classifier_type[index]))
        logging.info("  TRTS List of Accuracies: {} ".format(dict_metrics["TRTS accuracy"][classifier_type[index]]))
        logging.info("  TRTS List of Precisions: {} ".format(dict_metrics["TRTS precision"][classifier_type[index]]))
        logging.info("  TRTS List of Recalls: {} ".format(dict_metrics["TRTS recall"][classifier_type[index]]))
        logging.info("  TRTS List of F1-scores: {} ".format(dict_metrics["TRTS F1 score"][classifier_type[index]]))
        logging.info("  TRTS list log_likehood Score: {} ".format((dict_log_likehood_TRTS[classifier_type[index]])))
        logging.info("  TRTS list AUC: {} ".format((dict_TRTS_auc[classifier_type[index]])))
        logging.info("  TRTS Mean Accuracy: {} ".format(np.mean(dict_metrics["TRTS accuracy"][classifier_type[index]])))
        logging.info("  TRTS Mean Precision: {} ".format(np.mean(dict_metrics["TRTS precision"][classifier_type[index]])))
        logging.info("  TRTS Mean Recall: {} ".format(np.mean(dict_metrics["TRTS recall"][classifier_type[index]])))
        logging.info("  TRTS Mean F1 Score: {} ".format(np.mean(dict_metrics["TRTS F1 score"][classifier_type[index]])))
        logging.info("  TRTS Mean AUC: {} ".format(np.mean(dict_TRTS_auc[classifier_type[index]])))
        logging.info("  TRTS Mean log_likehood Score: {} ".format(np.mean(dict_log_likehood_TRTS[classifier_type[index]])))
        logging.info("  TRTS Standard Deviation of Accuracy: {} ".format(np.std(dict_metrics["TRTS accuracy"][classifier_type[index]])))
        logging.info("  TRTS Standard Deviation of Precision: {} ".format(np.std(dict_metrics["TRTS precision"][classifier_type[index]])))
        logging.info("  TRTS Standard Deviation of Recall: {} ".format(np.std(dict_metrics["TRTS recall"][classifier_type[index]])))
        logging.info("  TRTS Standard Deviation of F1 Score: {} \n".format(np.std(dict_metrics["TRTS F1 score"][classifier_type[index]])))
        logging.info("  TRTS Standard Deviation of log_likehood Score: {} ".format(np.std(dict_log_likehood_TRTS[classifier_type[index]])))
        logging.info("  TRTS Standard Deviation of AUC: {} ".format(np.std(dict_TRTS_auc[classifier_type[index]])))
        plot_filename = os.path.join(output_dir, f'{classifier_type[index]}_treinado_com_TSTR_testado_com_sint.pdf')

        plot_classifier_metrics.plot_classifier_metrics(classifier_type[index], dict_metrics["TRTS accuracy"][classifier_type[index]],
                                                        dict_metrics["TRTS precision"][classifier_type[index]], dict_metrics["TRTS recall"][classifier_type[index]],dict_metrics["TRTS F1 score"][classifier_type[index]],plot_filename,
                                                        f'{title_output_label}_TRTS',"TRTS")



       
        ac='TRTS Mean Accuracy '+f' Classifier Model {classifier_type[index]}'
        pc='TRTS Mean Precision '+f' Classifier Model {classifier_type[index]}'
        rr='TRTS Mean Recall '+f' Classifier Model {classifier_type[index]}'
        f1='TRTS Mean F1 Score '+f' Classifier Model {classifier_type[index]}'
        pct='TRTS Standard Deviation of Precision '+f'Classifier Model {classifier_type[index]}'
        act='TRTS Standard Deviation of Accuracy '+f' Classifier Model {classifier_type[index]}'
        rrt='TRTS Standard Deviation of Recall '+f' Classifier Model {classifier_type[index]}'
        f1t='TRTS Standard Deviation of f1 Score '+f' Classifier Model {classifier_type[index]}'
        if USE_AIM:
                values = {ac: np.mean(dict_metrics["TRTS accuracy"][classifier_type[index]]),
                  pc: np.mean(dict_metrics["TRTS precision"][classifier_type[index]]),
                  rr: np.mean(dict_metrics["TRTS recall"][classifier_type[index]]),
                  f1: np.mean(dict_metrics["TRTS F1 score"][classifier_type[index]]),
                  act: np.std(dict_metrics["TRTS accuracy"][classifier_type[index]]),
                  pct: np.std(dict_metrics["TRTS precision"][classifier_type[index]]),
                  rrt: np.std(dict_metrics["TRTS recall"][classifier_type[index]]),
                  f1t: np.std(dict_metrics["TRTS F1 score"][classifier_type[index]]),
                  }

                aim_run.track(values, context={'data': 'TRTS', 'classifier_type': classifier_type[index]})
               
        if USE_MLFLOW:
            values = {ac: np.mean(dict_metrics["TRTS accuracy"][classifier_type[index]]),
                  pc: np.mean(dict_metrics["TRTS precision"][classifier_type[index]]),
                  rr: np.mean(dict_metrics["TRTS recall"][classifier_type[index]]),
                  f1: np.mean(dict_metrics["TRTS F1 score"][classifier_type[index]]),
                  act: np.std(dict_metrics["TRTS accuracy"][classifier_type[index]]),
                  pct: np.std(dict_metrics["TRTS precision"][classifier_type[index]]),
                  rrt: np.std(dict_metrics["TRTS recall"][classifier_type[index]]),
                  f1t: np.std(dict_metrics["TRTS F1 score"][classifier_type[index]]),
                  }
            mlflow.log_metrics(values)

        if USE_TENSORBOARD:
            with file_writer.as_default():
                  tf.summary.scalar(ac, data=np.mean(dict_metrics["TRTS accuracy"][classifier_type[index]]),step=0)
                  tf.summary.scalar(pc, data=np.mean(dict_metrics["TRTS precision"][classifier_type[index]]),step=0)
                  tf.summary.scalar(rr, data=np.mean(dict_metrics["TRTS recall"][classifier_type[index]]),step=0)
                  tf.summary.scalar(f1, data=np.mean(dict_metrics["TRTS F1 score"][classifier_type[index]]),step=0)
                  tf.summary.scalar(act, data=np.std(dict_metrics["TRTS accuracy"][classifier_type[index]]),step=0)
                  tf.summary.scalar(pct, data=np.std(dict_metrics["TRTS precision"][classifier_type[index]]),step=0)
                  tf.summary.scalar(rrt, data=np.std(dict_metrics["TRTS recall"][classifier_type[index]]),step=0)
                  tf.summary.scalar(f1t, data=np.std(dict_metrics["TRTS F1 score"][classifier_type[index]]),step=0)

        #for analysis_type in zip(analysis_type_list):
        logging.info("Overall TSTR Results: {}\n".format(classifier_type[index]))
        logging.info("  TSTR List of Accuracies: {} ".format(dict_metrics["TSTR accuracy"][classifier_type[index]]))
        logging.info("  TSTR List of Precisions: {} ".format(dict_metrics["TSTR precision"][classifier_type[index]]))
        logging.info("  TSTR List of Recalls: {} ".format(dict_metrics["TSTR recall"][classifier_type[index]]))
        logging.info("  TSTR List of F1-scores: {} ".format(dict_metrics["TSTR F1 score"][classifier_type[index]]))
        logging.info("  TSTR list AUC: {} ".format((dict_TSTR_auc[classifier_type[index]])))
        logging.info("  TSTR list log_likehood Score: {} ".format((dict_log_likehood_TSTR[classifier_type[index]])))
        logging.info("  TSTR Mean Accuracy: {} ".format(np.mean(dict_metrics["TSTR accuracy"][classifier_type[index]])))
        logging.info("  TSTR Mean Precision: {} ".format(np.mean(dict_metrics["TSTR precision"][classifier_type[index]])))
        logging.info("  TSTR Mean Recall: {} ".format(np.mean(dict_metrics["TSTR recall"][classifier_type[index]])))
        logging.info("  TSTR Mean F1 Score: {} ".format(np.mean(dict_metrics["TSTR F1 score"][classifier_type[index]])))
        logging.info("  TSTR Mean log_likehood Score: {} ".format(np.mean(dict_log_likehood_TSTR[classifier_type[index]])))
        logging.info("  TSTR Mean AUC: {} ".format(np.mean(dict_TSTR_auc[classifier_type[index]])))
        logging.info("  TSTR Standard Deviation of Accuracy: {} ".format(np.std(dict_metrics["TSTR accuracy"][classifier_type[index]])))
        logging.info("  TSTR Standard Deviation of Precision: {} ".format(np.std(dict_metrics["TSTR precision"][classifier_type[index]])))
        logging.info("  TSTR Standard Deviation of Recall: {} ".format(np.std(dict_metrics["TSTR recall"][classifier_type[index]])))
        logging.info("  TSTR Standard Deviation of F1 Score: {} \n".format(np.std(dict_metrics["TSTR F1 score"][classifier_type[index]])))
        logging.info("  TSTR Standard Deviation of log_likehood Score: {} ".format(np.std(dict_log_likehood_TSTR[classifier_type[index]])))
        logging.info("  TSTR Standard Deviation of AUC: {} ".format(np.std(dict_TSTR_auc[classifier_type[index]])))
        plot_filename = os.path.join(output_dir, f'{classifier_type[index]}_treiando_com_sint_testado_com_TSTR.pdf')

        plot_classifier_metrics.plot_classifier_metrics(classifier_type[index], dict_metrics["TSTR accuracy"][classifier_type[index]],
                                                        dict_metrics["TSTR precision"][classifier_type[index]] ,dict_metrics["TSTR recall"][classifier_type[index]],
                                                        dict_metrics["TSTR F1 score"][classifier_type[index]] ,plot_filename,
                                                        f'{title_output_label}_TSTR',"TSTR")

        ac='TSTR Mean Accuracy '+f' Classifier Model {classifier_type[index]}'
        pc='TSTR Mean Precision '+f' Classifier Model {classifier_type[index]}'
        rr='TSTR Mean Recall '+f' Classifier Model {classifier_type[index]}'
        f1='TSTR Mean f1 Score '+f' Classifier Model {classifier_type[index]}'
        pct='TSTR Standart Deviation of PreCision '+f'Classifier Model {classifier_type[index]}'
        act='TSTR Standard Deviation of Accuracy '+f' Classifier Model {classifier_type[index]}'
        rrt='TSTR Standard Deviation of Recall '+f' Classifier Model {classifier_type[index]}'
        f1t='TSTR Standard Deviation of F1 Score '+f' Classifier Model {classifier_type[index]}'
        if USE_AIM:
           aim_run.track(values,context={'data': 'TSTR','classifir_type': classifier_type[index]})

        if USE_MLFLOW:
            mlflow.log_metrics(values)
           # d = Distribution(TSTR_accuracies[index])

#            mlflow.log_metric('accuracy_dist',TSTR_accuracies[index],step=index)

        if USE_TENSORBOARD:
            with file_writer.as_default():
                  tf.summary.scalar(ac, data=np.mean(dict_metrics["TSTR accuracy"][classifier_type[index]]),step=0)
                  tf.summary.scalar(pc, data=np.mean(dict_metrics["TSTR precision"][classifier_type[index]]),step=0)
                  tf.summary.scalar(rr, data=np.mean(dict_metrics["TSTR recall"][classifier_type[index]]),step=0)
                  tf.summary.scalar(f1, data=np.mean(dict_metrics["TSTR F1 score"][classifier_type[index]]),step=0)
                  tf.summary.scalar(act, data=np.std(dict_metrics["TSTR accuracy"][classifier_type[index]]),step=0)
                  tf.summary.scalar(pct, data=np.std(dict_metrics["TSTR precision"][classifier_type[index]]),step=0)
                  tf.summary.scalar(rrt, data=np.std(dict_metrics["TSTR recall"][classifier_type[index]]),step=0)
                  tf.summary.scalar(f1t, data=np.std(dict_metrics["TSTR F1 score"][classifier_type[index]]),step=0)     
    comparative_metrics = ['Mean Squared Error', 'Cosine Similarity', 'Max Mean Discrepancy']


    comparative_lists = ["list_mean_squared_error","list_cosine_similarity","list_maximum_mean_discrepancy"]
    logging.info(f"Comparative Metrics:")
    for metric, comparative_list in zip(comparative_metrics,comparative_lists):
        logging.info("\t{}".format(metric))
        for i in ["false","positive"]:
          logging.info("\t\t {}".format(i))
          logging.info("\t\t{} - List     : {}".format(metric, dict_similarity[comparative_list][i]))
          logging.info("\t\t{} - Mean    : {}".format(metric, np.mean(dict_similarity[comparative_list][i])))
          logging.info("\t\t{} - Std. Dev.  : {}\n".format(metric, np.std(dict_similarity[comparative_list][i])))
    
    p_acc=0
    p_prec=0
    p_f1=0
    p_recall=0
    for index in range(len(classifier_type)):
     p_acc=p_value_test(dict_metrics["TSTR accuracy"],dict_metrics["TRTS accuracy"],"accuracy",classifier_type[index])
     p_prec=p_value_test(dict_metrics["TSTR precision"],dict_metrics["TRTS precision"],"precision",classifier_type[index])
     p_f1=p_value_test(dict_metrics["TSTR F1 score"],dict_metrics["TRTS F1 score"],"F1 score",classifier_type[index])
     p_recall=p_value_test(dict_metrics["TSTR recall"],dict_metrics["TRTS recall"],"recall",classifier_type[index])
     p_auc=p_value_test(dict_TSTR_auc,dict_TRTS_auc,"auc",classifier_type[index])
     plot_filename = os.path.join(output_dir, f'{classifier_type[index]}_p_values.pdf')

    plot_filename1 = os.path.join(output_dir, f'Comparison_TSTR_TRTS_positive.jpg')
    plot_filename2 = os.path.join(output_dir, f'Comparison_TSTR_TRTS_false.jpg')

    
    plot_regressive_metrics.plot_regressive_metrics(dict_similarity["list_mean_squared_error"]["positive"],dict_similarity["list_cosine_similarity"]["positive"],dict_similarity["list_maximum_mean_discrepancy"]["positive"],plot_filename1,f'{title_output_label}')
    plot_regressive_metrics.plot_regressive_metrics(dict_similarity["list_mean_squared_error"]["false"],dict_similarity["list_cosine_similarity"]["false"],dict_similarity["list_maximum_mean_discrepancy"]["false"],plot_filename2,f'{title_output_label}')
    if USE_AIM:
       aim_image = Image(plot_filename1)
       aim_image2 = Image(plot_filename2)
       aim_run.track(value=aim_image, name=plot_filename1,context={'classifier_type': classifier_type[index]})
       aim_run.track(value=aim_image2, name=plot_filename2,context={'classifier_type': classifier_type[index]})
    if USE_TENSORBOARD:
         with file_writer.as_default():
                 cm_image = plot_to_image(plot_filename)
                 tf.summary.image('Comparison_TSTR_TRTS.jpg', cm_image,step=0)
    if USE_MLFLOW:
       mlflow.log_artifact(plot_filename1,'images')
       mlflow.log_artifact(plot_filename2,'images')
 




 
def get_adversarial_model(latent_dim, input_data_shape, activation_function, initializer_mean, initializer_deviation,
                          dropout_decay_rate_g, dropout_decay_rate_d, last_layer_activation, dense_layer_sizes_g,
                          dense_layer_sizes_d, dataset_type, training_algorithm, latent_mean_distribution,
                          latent_stander_deviation):
    instance_models = ConditionalGAN(latent_dim, input_data_shape, activation_function, initializer_mean,
                                     initializer_deviation, dropout_decay_rate_g, dropout_decay_rate_d,
                                     last_layer_activation, dense_layer_sizes_g, dense_layer_sizes_d, dataset_type)

    generator_model = instance_models.get_generator()
    discriminator_model = instance_models.get_discriminator()

    adversarial_model = AdversarialModel(generator_model, discriminator_model, latent_dimension=latent_dim,
                                         latent_mean_distribution=latent_mean_distribution,
                                         latent_stander_deviation=latent_stander_deviation)
    
   # model(adversarial_model)

    optimizer_generator = adversarial_model.get_optimizer(training_algorithm)
    optimizer_discriminator = adversarial_model.get_optimizer(training_algorithm)
    loss_generator = BinaryCrossentropy()
    loss_discriminator = BinaryCrossentropy()

    adversarial_model.compile(optimizer_generator, optimizer_discriminator, loss_generator, loss_discriminator,  loss=keras.losses.SparseCategoricalCrossentropy(),)
    return adversarial_model


def show_model(latent_dim, input_data_shape, activation_function, initializer_mean,
               initializer_deviation, dropout_decay_rate_g, dropout_decay_rate_d,
               last_layer_activation, dense_layer_sizes_g, dense_layer_sizes_d,
               dataset_type, verbose_level):
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

    show_model(latent_dim, input_data_shape, activation_function, initializer_mean,
               initializer_deviation, dropout_decay_rate_g, dropout_decay_rate_d,
               last_layer_activation, dense_layer_sizes_g, dense_layer_sizes_d,
               dataset_type, verbose_level)

    stratified = StratifiedKFold(n_splits=k, shuffle=True)

  
    dict_similarity={"list_mean_squared_error":{"positive":[],"false":[]},
    "list_cosine_similarity":{"positive":[],"false":[]},
    "list_kl_divergence":{"positive":[],"false":[]},
    "list_maximum_mean_discrepancy":{"positive":[],"false":[]}}
    dict_log_likehood_TRTS={"RandomForest":[],"AdaBoost":[],"DecisionTree":[], "Perceptron":[],"SVM":[],"SGDRegressor":[],"XGboost":[]}
    dict_log_likehood_TSTR={"RandomForest":[],"AdaBoost":[],"DecisionTree":[], "Perceptron":[],"SVM":[],"SGDRegressor":[],"XGboost":[]}
    dict_TRTS_auc={"RandomForest":[],"AdaBoost":[],"DecisionTree":[], "Perceptron":[],"SVM":[],"SGDRegressor":[],"XGboost":[]}
    dict_TSTR_auc={"RandomForest":[],"AdaBoost":[],"DecisionTree":[], "Perceptron":[],"SVM":[],"SGDRegressor":[],"XGboost":[]}
    dict_metrics={"TSTR accuracy":{"RandomForest":[],"AdaBoost":[],"DecisionTree":[], "Perceptron":[],"SVM":[],"SGDRegressor":[],"XGboost":[]},
         "TRTS accuracy":{"RandomForest":[],"AdaBoost":[],"DecisionTree":[], "Perceptron":[],"SVM":[],"SGDRegressor":[],"XGboost":[]},
         "TSTR precision":{"RandomForest":[],"AdaBoost":[],"DecisionTree":[], "Perceptron":[],"SVM":[],"SGDRegressor":[],"XGboost":[]},
         "TRTS precision":{"RandomForest":[],"AdaBoost":[],"DecisionTree":[], "Perceptron":[],"SVM":[],"SGDRegressor":[],"XGboost":[]},
         "TSTR F1 score":{"RandomForest":[],"AdaBoost":[],"DecisionTree":[], "Perceptron":[],"SVM":[],"SGDRegressor":[],"XGboost":[]},
         "TRTS F1 score":{"RandomForest":[],"AdaBoost":[],"DecisionTree":[],"Perceptron":[],"SVM":[],"SGDRegressor":[],"XGboost":[]},
         "TSTR recall":{"RandomForest":[],"AdaBoost":[],"DecisionTree":[], "Perceptron":[],"SVM":[],"SGDRegressor":[],"XGboost":[]},
         "TRTS recall":{"RandomForest":[],"AdaBoost":[],"DecisionTree":[], "Perceptron":[],"SVM":[],"SGDRegressor":[],"XGboost":[]}}
   #list_of_p={}
    for i, (train_index, test_index) in enumerate(stratified.split(dataset.iloc[:, :-1], dataset.iloc[:, -1])):
        adversarial_model = get_adversarial_model(latent_dim, input_data_shape, activation_function, initializer_mean,
                                                  initializer_deviation, dropout_decay_rate_g, dropout_decay_rate_d,
                                                  last_layer_activation, dense_layer_sizes_g, dense_layer_sizes_d,
                                                  dataset_type, training_algorithm, latent_mean_distribution,
                                                  latent_stander_deviation)


        instance_classifier_TRTS = Classifiers()
        instance_classifier_TSTR = Classifiers()
        x_training = np.array(dataset.iloc[train_index, :-1].values, dtype=dataset_type)
        x_test = np.array(dataset.iloc[test_index, :-1].values, dtype=dataset_type)
        y_training = np.array(dataset.iloc[train_index, -1].values, dtype=dataset_type)
        y_test = np.array(dataset.iloc[test_index, -1].values, dtype=dataset_type)
        x_training = x_training[0:int(len(x_training) - (len(x_training) % batch_size)), :]
        y_training = y_training[0:int(len(y_training) - (len(y_training) % batch_size))]

        logging.info(f" Starting training ADVERSARIAL MODEL:\n")
        if USE_TENSORBOARD:
          training_history = adversarial_model.fit(x_training, y_training, epochs=number_epochs, batch_size=batch_size,
                                                 verbose=DEFAULT_VERBOSE_LIST[verbose_level],callbacks=callbacks)
        else:
          training_history = adversarial_model.fit(x_training, y_training, epochs=number_epochs, batch_size=batch_size,
                                                 verbose=DEFAULT_VERBOSE_LIST[verbose_level])
        logging.info(f"     Finished training\n")
        
        if save_models:
            adversarial_model.save_models(output_dir, i)

        generator_loss_list = training_history.history['loss_g']
        discriminator_loss_list = training_history.history['loss_d']
        plot_loss_curve_instance = PlotCurveLoss()
        plot_loss_curve_instance.plot_training_loss_curve(generator_loss_list, discriminator_loss_list, output_dir, i,
                                                          path_curve_loss)

        number_samples_true = len([positional_label for positional_label in y_test.tolist() if positional_label == 1])
        number_samples_false = len([positional_label for positional_label in y_test.tolist() if positional_label == 0])

        # Calcular o número desejado de amostras sintéticas para cada classe
        num_samples_true_desired = len([positional_label for positional_label in y_training.tolist() if positional_label == 1])
        num_samples_false_desired = len([positional_label for positional_label in y_training.tolist() if positional_label == 0])
        print(num_samples_true_desired)
        print(num_samples_false_desired)
        # Gerar dados sintéticos para a classe verdadeira (label 1)
        x_true_synthetic, y_true_synthetic = generate_samples(adversarial_model, num_samples_true_desired, latent_dim,
                                                              1, verbose_level, latent_mean_distribution,
                                                              latent_stander_deviation)

        # Gerar dados sintéticos para a classe falsa (label 0)
        x_false_synthetic, y_false_synthetic = generate_samples(adversarial_model, num_samples_false_desired,
                                                                latent_dim,
                                                                0, verbose_level, latent_mean_distribution,
                                                                latent_stander_deviation)

        # Juntar os dados sintéticos gerados para ambas as classes
        x_synthetic_samples = np.concatenate((x_true_synthetic, x_false_synthetic), axis=0)
        y_synthetic_samples = np.rint(np.concatenate((y_true_synthetic, y_false_synthetic), axis=0))
        y_synthetic_samples = np.squeeze(y_synthetic_samples, axis=1)

        # Converter para DataFrame e adicionar os nomes das colunas
        synthetic_columns = dataset.columns[:-1]
        df_synthetic = pd.DataFrame(data=x_synthetic_samples, columns=synthetic_columns)
        df_synthetic['class'] = y_synthetic_samples

            # Salvar dados sintéticos em um arquivo CSV
        synthetic_filename = f'synthetic_data_fold_{i + 1}.csv'
        synthetic_filepath = os.path.join(output_dir, synthetic_filename)
        df_synthetic.to_csv(synthetic_filepath, index=False, sep=',', header=True)

        #TRTS DATASET
        x_synthetic_training=x_synthetic_samples
        y_synthetic_training=y_synthetic_samples
        print(x_synthetic_training.shape)
        print(y_synthetic_training.shape)
        if USE_TENSORBOARD:
          adversarial_model.fit(x_test, y_test)
        else:
          adversarial_model.fit(x_test, y_test)

        # Generate synthetic samples with the desired number for each class
        x_true_synthetic, y_true_synthetic = generate_samples(adversarial_model, number_samples_true, latent_dim,
                                                            1, verbose_level, latent_mean_distribution,
                                                              latent_stander_deviation)

        x_false_synthetic, y_false_synthetic = generate_samples(adversarial_model, number_samples_false, latent_dim,
                                                             0, verbose_level, latent_mean_distribution,
                                                              latent_stander_deviation)
        # Ensure the desired number of samples for each class


        x_synthetic_samples = np.concatenate((x_true_synthetic, x_false_synthetic), dtype=dataset_type)
        y_synthetic_samples = np.rint(np.concatenate((y_true_synthetic, y_false_synthetic)))

        y_synthetic_samples = np.squeeze(y_synthetic_samples, axis=1)
     

        list_classifiers_TRTS = instance_classifier_TRTS.get_trained_classifiers(classifier_list, x_training, y_training,
                                                                       dataset_type, verbose_level, input_data_shape)
                                                                       
        list_classifiers_TSTR = instance_classifier_TSTR.get_trained_classifiers(classifier_list, x_synthetic_training, y_synthetic_training,
                                                                       dataset_type, verbose_level, input_data_shape)    
                                                                       
        evaluation_results_synthetic_data = evaluate_TRTS_data(list_classifiers_TRTS, x_synthetic_samples,
                                                                    y_synthetic_samples, i, k, True,
                                                                    output_dir, classifier_list, title_output,
                                                                    path_confusion_matrix, verbose_level,dict_metrics,dict_TRTS_auc,dict_log_likehood_TRTS)
        evaluation_results_real_data = evaluate_TSTR_data(list_classifiers_TSTR, x_test, y_test, i, k,
                                                          True, output_dir, classifier_list,
                                                          title_output, path_confusion_matrix, verbose_level,dict_metrics,dict_TSTR_auc,dict_log_likehood_TSTR)
                             
        uni,counts=np.unique(y_test,return_counts=True)
        indexation=0
        if counts[1]>counts[0]:
            indexation=counts[0]
        else:
            indexation=counts[1]
        falses=x_test[y_test==0]
        falses=falses[:indexation]
        positives=x_test[y_test==1]
        positives=positives[:indexation]
        x_false_synthetic=x_false_synthetic[:indexation]
        x_true_synthetic=x_true_synthetic[:indexation]
        comparative_metrics1 = comparative_data(i,x_false_synthetic, falses,"falses")
        comparative_metrics2 = comparative_data(i,x_true_synthetic,positives,"true")

        #comparative_metrics = comparative_data(i, x_synthetic_samples, x_test)

        dict_similarity["list_mean_squared_error"]["false"].append(comparative_metrics1[0])
        dict_similarity["list_cosine_similarity"]["false"].append(comparative_metrics1[1])
        dict_similarity["list_maximum_mean_discrepancy"]["false"].append(comparative_metrics1[2])
      
        dict_similarity["list_mean_squared_error"]["positive"].append(comparative_metrics2[0])
        dict_similarity["list_cosine_similarity"]["positive"].append(comparative_metrics2[1])
        dict_similarity["list_maximum_mean_discrepancy"]["positive"].append(comparative_metrics2[2])
    show_and_export_results(dict_similarity,classifier_list, output_dir, title_output,dict_metrics,dict_TRTS_auc,dict_TSTR_auc,dict_log_likehood_TSTR,dict_log_likehood_TRTS)

def initial_step(initial_arguments, dataset_type):
    
    file_args = os.path.join(initial_arguments.output_dir, 'commandline_args.txt')

    with open(file_args, 'w') as f:
        json.dump(initial_arguments.__dict__, f, indent=2)

    dataset_file_loaded = pd.read_csv(initial_arguments.input_dataset, dtype=dataset_type)
    dataset_file_loaded = dataset_file_loaded.dropna()
    dataset_input_shape = dataset_file_loaded.shape[1] - 1

    input_dataset = os.path.basename(initial_arguments.input_dataset)
    dataset_labels = f'Dataset: {input_dataset}'
    return dataset_file_loaded, dataset_input_shape, dataset_labels


def show_all_settings(arg_parsers):
    logging.info("Command:\n\t{0}\n".format(" ".join([x for x in sys.argv])))
    logging.info("Settings:")
    lengths = [len(x) for x in vars(arg_parsers).keys()]
    max_length = max(lengths)

    for k, v in sorted(vars(arg_parsers).items()):
        settings_parser = "\t"
        settings_parser += k.ljust(max_length, " ")
        settings_parser += " : {}".format(v)
        logging.info(settings_parser)

    logging.info("")


def create_argparse():

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

    parser.add_argument('--num_samples_class_malware', type=int,
                        default=DEFAULT_NUMBER_GENERATE_MALWARE_SAMPLES,
                        help='Número de amostras da Classe 1 (maligno).')

    parser.add_argument('--num_samples_class_benign', type=int,
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



    arguments = create_argparse()

    logging_format = '%(asctime)s\t***\t%(message)s'

    # configura o mecanismo de logging
    if arguments.verbosity == logging.DEBUG:
        # mostra mais detalhes
        logging_format = '%(asctime)s\t***\t%(levelname)s {%(module)s} [%(funcName)s] %(message)s'

    Path(arguments.output_dir).mkdir(parents=True, exist_ok=True)
    logging_filename = os.path.join(arguments.output_dir, LOGGING_FILE_NAME)

    # formatter = logging.Formatter(logging_format, datefmt=TIME_FORMAT, level=arguments.verbosity)
    logging.basicConfig(format=logging_format, level=arguments.verbosity)

    # Add file rotating handler, with level DEBUG
    rotatingFileHandler = RotatingFileHandler(filename=logging_filename, maxBytes=100000, backupCount=5)
    rotatingFileHandler.setLevel(arguments.verbosity)
    rotatingFileHandler.setFormatter(logging.Formatter(logging_format))
    logging.getLogger().addHandler(rotatingFileHandler)
    
    show_all_settings(arguments)
    

    time_start_campaign = datetime.datetime.now()

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
    if USE_AIM:
        output_dir = arguments.output_dir
        experiment_name= output_dir.split('/')[-1]
        aim_run=Run(experiment=experiment_name)
    if USE_MLFLOW:
       mlflow.set_tracking_uri("http://127.0.0.1:6000/")
       
       if arguments.run_id==None:
             
             mlflow.start_run()
             mlflow.set_experiment(arguments.output_dir)
       else:
             mlflow.start_run(run_id=arguments.run_id)
    if USE_TENSORBOARD:
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

    time_end_campaign = datetime.datetime.now()
    if USE_TENSORBOARD:
       file_writer.close()
