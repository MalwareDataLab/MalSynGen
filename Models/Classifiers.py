"""
Este módulo define os métodos utilizados para instanciar e treinar os classificadores utilizados durante a execução.
Classes: 
- Classifiers: A classe Classifiers é responsável por criar, configurar e treinar diversos modelos de classificação utilizando parâmetros especificados pelo usuário ou valores padrão. 
A classe suporta múltiplos algoritmos de aprendizado supervisionado, incluindo:
                -Random Forest.
                -K-Nearest Neighbors.
                -Support Vector Machine. 
                -Gaussian Process.
                -Decision Tree.
                -AdaBoost.
                -Naive Bayes.
                -Quadratic Discriminant Analysis.
                -Perceptron Multilayer.
                -XGBoost.
                -SGD Regressor.
"""
# Importação de bibliotecas necessárias
import logging
import numpy as np
from tensorflow import keras
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDRegressor
from Models.PerceptronModel import PerceptronMultilayer
import xgboost as xgb
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

# Parâmetros padrão para cada tipo de classificador
## Valor padrão para o número de estimadores para o Random Forest.
DEFAULT_RANDOM_FOREST_NUMBER_ESTIMATORS = 100
## Valor padrão para a profundidade do Random Forest
DEFAULT_RANDOM_FOREST_MAX_DEPTH = None
## Valor padrão para o número maximo de folhas para o Random forest
DEFAULT_RANDOM_FOREST_MAX_LEAF_NODES = None

## Valor padrão de regularização para o SVM
DEFAULT_SUPPORT_VECTOR_MACHINE_REGULARIZATION = 1.0
## Kernel padrão para a SVM
DEFAULT_SUPPORT_VECTOR_MACHINE_KERNEL = "rbf"
## Valor padrão o número máximo de iterações para o processo gaussiano
DEFAULT_SUPPORT_VECTOR_MACHINE_KERNEL_DEGREE = 3
##  Valor padrão para o coeficiente gamma para SVM
DEFAULT_SUPPORT_VECTOR_MACHINE_GAMMA = "scale"
##Valor padrão para o número de viznhos para O KNN
DEFAULT_KNN_NUMBER_NEIGHBORS = 5
## Função de peso padrão para o KNN.
DEFAULT_KNN_WEIGHTS = "uniform"
## Algortimo padrão para o KNN
DEFAULT_KNN_ALGORITHM = "auto"
## Valor padrão para folhas para o KNN
DEFAULT_KNN_LEAF_SIZE = 30
## Métrica padrão utilizada para a computação da distância para o KNN
DEFAULT_KNN_METRIC = "minkowski"
## Kernel padrão para o processo gaussiano
DEFAULT_GAUSSIAN_PROCESS_KERNEL = None
## Valor padrõa para o número máximo de iterações para o processo gaussiano
DEFAULT_GAUSSIAN_PROCESS_MAX_ITERATIONS = 20
##  Otimizador padrão para o processo gaussiano
DEFAULT_GAUSSIAN_PROCESS_OPTIMIZER = "fmin_l_bfgs_b"

## Critério padrão usado para medir a qualidade da divisão na árvore de decisão
DEFAULT_DECISION_TREE_CRITERION = "gini"
## Valor padrão para a profundidade máxima da árvore de decisão
DEFAULT_DECISION_TREE_MAX_DEPTH = None
## Valor padrão para o número máximo de características a serem consideradas
DEFAULT_DECISION_TREE_MAX_FEATURE = None
## Valor padrão para o número máximo de folhas na árvore de decisão
DEFAULT_DECISION_TREE_MAX_LEAF = None

##Estimador base padrão para o AdaBoost
DEFAULT_ADA_BOOST_ESTIMATOR = None
## Valor padrão para o  número maximo de estimadores para o AdaBoost
DEFAULT_ADA_BOOST_NUMBER_ESTIMATORS = 50
## Valor padrão para a taxa de aprendizado do AdaBosst
DEFAULT_ADA_BOOST_LEARNING_RATE = 1.0
## Algoritmo padrão utilizado pelo AdaBoost
DEFAULT_ADA_BOOST_ALGORITHM = "SAMME.R"

## Valor padrão para a probabilidades de Priors para Naive Bayes
DEFAULT_NAIVE_BAYES_PRIORS = None
## Valor padrão para o parâmetro de suavização para Naive Bayes
DEFAULT_NAIVE_BAYES_VARIATION_SMOOTHING = 1e-09
## Valor padrão para a probabilidades de priors para QDA.
DEFAULT_QUADRATIC_DISCRIMINANT_ANALYSIS_PRIORS = None
## Valor padrão para o parâmetro de regularização para QDA.
DEFAULT_QUADRATIC_DISCRIMINANT_ANALYSIS_REGULARIZATION = 0.0
## Valor padrão para o parâmetro de regularização para QDA
DEFAULT_QUADRATIC_DISCRIMINANT_THRESHOLD = 0.0001

## Algoritmo padrão de treinamento para Perceptron.
DEFAULT_PERCEPTRON_TRAINING_ALGORITHM = "Adam"
## Função de perda padrão para o perceptron
DEFAULT_PERCEPTRON_LOSS = "binary_crossentropy"
##  Valores padrões para as camadas do perceptron
DEFAULT_PERCEPTRON_LAYERS_SETTINGS = [512, 256, 256]
## Valor padrão para a taxa de dropout do perceptron
DEFAULT_PERCEPTRON_DROPOUT_DECAY_RATE = 0.2
## Métrica padrão utilizada pelo perceptron para avaliar sua performance
DEFAULT_PERCEPTRON_METRIC = ["accuracy"]
## Função padrão de ativação para as camadas internas
DEFAULT_PERCEPTRON_LAYER_ACTIVATION = keras.activations.swish
## Função padrão de ativação para a última camada
DEFAULT_PERCEPTRON_LAST_LAYER_ACTIVATION = "sigmoid"
## Valor padrão paro número de épocas para o perceptron
DEFAULT_PERCEPTRON_NUMBER_EPOCHS = 1



DEFAULT_CLASSIFIER_LIST = ["RandomForest", "SupportVectorMachine", "KNN",
                           "GaussianPrecess", "DecisionTree", "AdaBoost",
                           "NaiveBayes", "QuadraticDiscriminant", "Perceptron","SGDRegressor","XGboost"]
DEFAULT_VERBOSE_LIST = {logging.INFO: 2, logging.DEBUG: 1, logging.WARNING: 2,
                        logging.FATAL: 0, logging.ERROR: 0}


class Classifiers:
    """
A classe Classifiers é responsável por criar, configurar e treinar diversos modelos de classificação utilizando parâmetros especificados pelo usuário ou valores padrão. 
A classe suporta múltiplos algoritmos de aprendizado supervisionado, incluindo:
                -Random Forest.
                -K-Nearest Neighbors.
                -Support Vector Machine. 
                -Gaussian Process.
                -Decision Tree.
                -AdaBoost.
                -Naive Bayes.
                -Quadratic Discriminant Analysis.
                -Perceptron Multilayer.
                -XGBoost.
                -SGD Regressor.
    Funções:
        - __init__ : Inicializa a classe Classifiers com os parâmetros padrão ou especificados.
        - _get_instance_random_forest : Retorna uma instância treinada de um classificador RandomForest.
        - __get_instance_k_neighbors_classifier :  Retorna uma instância treinada de um classificador KNN.
        - __get_instance_support_vector_machine :  Retorna uma instância treinada de um classifcador SVM.
        - __get_instance_gaussian_process : Retorna uma instância treinada de um classificador de processo Gausiano.
        - __get_instance_xgboost : Retorna uma instância treinada de um classificador XGBoost.
        - __get_instance_SGDRegressor : Retorna uma instância treinada de um classificador SGD.
        - __get_instance_decision_tree : Retorna uma instância treinada de um classificador Decision Tree.
        - __get_instance_ada_boost : Retorna uma instância treinada de um classificador  AdaBoost.
        - __get_instance_naive_bayes : Retorna uma instância treinada de um classificador  Naive Bayes.
        - __get_instance_quadratic_discriminant : Retorna uma instância treinada de um classificador discriminador quadrático.
        - __get_instance_perceptron : Retorna uma instância treinada de um classificador perceptron de multi camadas.
        - get_trained_classifiers : Retorna uma lista de classificadores treinados baseados na lista de classificadores fornecida.
        - set_random_forest_number_estimators : Define o número de estimadores para o Random Forest.
        - set_random_forest_max_depth :  Define a profundidade máxima da árvore do algoritmo Random Forest.
        - max_leaf_nodes :   Define o número máximo de folhas no Random Forest.
        - set_knn_number_neighbors : Define o número de vizinhos para o algoritmo KNN.
        - set_knn_weights : Define a função de peso para o algoritmo KNN.
        - set_knn_leaf_size : Define o tamanho da folha para o KNN.
        - set_knn_metric : Define a  métrica utilizada para a computação da distância para o KNN.
        - set_knn_algorithm : Define o algoritmo usado pelo KNN.
        - set_support_vector_machine_normalization: Define o parâmetro de regularização para SVM.
        - set_support_vector_machine_kernel : Define o kernel usado pelo SVM.
        - set_support_vector_machine_kernel_degree : Define o grau do kernel para SVM.
        - set_support_vector_machine_gamma : Define o coeficiente gamma para SVM.
        - set_gaussian_process_kernel: Define  o kernel para o processo gaussiano.
        - set_gaussian_process_max_iterations : Define  o número máximo de iterações para o processo gaussiano.
        - set_gaussian_process_optimizer : Define  o otimizador para o processo gaussiano.
        - set_decision_tree_criterion : Define o critério usado para medir a qualidade da divisão na árvore de decisão.
        - set_decision_tree_max_depth : Define a profundidade máxima da árvore de decisão.
        - set_decision_tree_max_feature : Define o número máximo de características a serem consideradas para a árvore de decisão.
        - set_decision_tree_max_leaf : Define o número máximo de folhas na árvore de decisão.
        - set_ada_boost_estimator : Define o estimador base para AdaBoost.
        - set_ada_boost_number_estimators : Define o  número maximo de estimadores para o AdaBoost.
        - set_ada_boost_learning_rate : Define a taxa de aprendizado do AdaBoost.
        - set_ada_boost_algorithm : Define o algoritmo a utilizado pelo AdaBoost.
        - set_naive_bayes_priors : Define a probabilidades de Priors para Naive Bayes.
        - set_naive_bayes_variation_smoothing : Define o parâmetro de suavização para Naive Bayes.
        - set_quadratic_discriminant_analysis_priors : Define  a probabilidades de priors para QDA.
        - set_quadratic_discriminant_analysis_regularization : Define o parâmetro de regularização para QDA.
        - set_perceptron_training_algorithm : Define o algoritmo de treinamento para Perceptron.
        - set_perceptron_training_loss : Define  a função de perda para o Perceptron.
        - set_perceptron_layer_activation : Define a função de ativação das camadas internas do Perceptron.
    """

    def __init__(self, random_forest_number_estimators=DEFAULT_RANDOM_FOREST_NUMBER_ESTIMATORS,
                 random_forest_max_depth=DEFAULT_RANDOM_FOREST_MAX_DEPTH,
                 max_leaf_nodes=DEFAULT_RANDOM_FOREST_MAX_LEAF_NODES, knn_number_neighbors=DEFAULT_KNN_NUMBER_NEIGHBORS,
                 knn_weights=DEFAULT_KNN_WEIGHTS, knn_leaf_size=DEFAULT_KNN_LEAF_SIZE, knn_metric=DEFAULT_KNN_METRIC,
                 knn_algorithm=DEFAULT_KNN_ALGORITHM, support_vector_machine_gamma=DEFAULT_SUPPORT_VECTOR_MACHINE_GAMMA,
                 support_vector_machine_normalization=DEFAULT_SUPPORT_VECTOR_MACHINE_REGULARIZATION,
                 support_vector_machine_kernel=DEFAULT_SUPPORT_VECTOR_MACHINE_KERNEL,
                 support_vector_machine_kernel_degree=DEFAULT_SUPPORT_VECTOR_MACHINE_KERNEL_DEGREE,
                 gaussian_process_kernel=DEFAULT_GAUSSIAN_PROCESS_KERNEL,
                 gaussian_process_max_iterations=DEFAULT_GAUSSIAN_PROCESS_MAX_ITERATIONS,
                 gaussian_process_optimizer=DEFAULT_GAUSSIAN_PROCESS_OPTIMIZER,
                 decision_tree_criterion=DEFAULT_DECISION_TREE_CRITERION,
                 decision_tree_max_depth=DEFAULT_DECISION_TREE_MAX_DEPTH,
                 decision_tree_max_feature=DEFAULT_DECISION_TREE_MAX_FEATURE,
                 decision_tree_max_leaf=DEFAULT_DECISION_TREE_MAX_LEAF, ada_boost_estimator=DEFAULT_ADA_BOOST_ESTIMATOR,
                 ada_boost_number_estimators=DEFAULT_ADA_BOOST_NUMBER_ESTIMATORS,
                 ada_boost_learning_rate=DEFAULT_ADA_BOOST_LEARNING_RATE,
                 ada_boost_algorithm=DEFAULT_ADA_BOOST_ALGORITHM, naive_bayes_priors=DEFAULT_NAIVE_BAYES_PRIORS,
                 naive_bayes_variation_smoothing=DEFAULT_NAIVE_BAYES_VARIATION_SMOOTHING,
                 quadratic_discriminant_analysis_priors=DEFAULT_QUADRATIC_DISCRIMINANT_ANALYSIS_PRIORS,
                 quadratic_discriminant_analysis_regularization=DEFAULT_QUADRATIC_DISCRIMINANT_ANALYSIS_REGULARIZATION,
                 quadratic_discriminant_threshold=DEFAULT_QUADRATIC_DISCRIMINANT_THRESHOLD,
                 perceptron_training_algorithm=DEFAULT_PERCEPTRON_TRAINING_ALGORITHM,
                 perceptron_training_loss=DEFAULT_PERCEPTRON_LOSS, perceptron_training_metric=None,
                 perceptron_layer_activation=DEFAULT_PERCEPTRON_LAYER_ACTIVATION,
                 perceptron_last_layer_activation=DEFAULT_PERCEPTRON_LAST_LAYER_ACTIVATION,
                 perceptron_dropout_decay_rate=DEFAULT_PERCEPTRON_DROPOUT_DECAY_RATE,
                 perceptron_number_epochs=DEFAULT_PERCEPTRON_NUMBER_EPOCHS,
                 perceptron_layers_settings=None):
        """
        Inicializa a classe Classifiers com os parâmetros padrão ou especificados.
        
        Parâmetros:
           - random_forest_number_estimators : Número de estimadores para o Random Forest.
           - random_forest_max_depth : Profundidade máxima da árvore no Random Forest.
           - max_leaf_nodes : Número máximo de folhas no Random Forest.
           - knn_number_neighbors : Número de vizinhos para o KNN.
           - knn_weights : Função de peso para o KNN.
           - knn_leaf_size : Tamanho da folha para o KNN.
           - knn_metric : Métrica utilizada para a computação da distância para o KNN. 
           - knn_algorithm: Algoritmo usado pelo KNN.
           - support_vector_machine_gamma:  Coeficiente gamma para SVM.
           - support_vector_machine_normalization: Parâmetro de regularização para SVM.
           - support_vector_machine_kernel : Kernel usado pelo SVM.
           - support_vector_machine_kernel_degree : Grau do kernel para SVM.
           - gaussian_process_kernel : Kernel para o processo gaussiano.
           - gaussian_process_max_iterations : Número máximo de iterações para o processo gaussiano.
           - gaussian_process_optimizer : Otimizador para o processo gaussiano.
           - decision_tree_criterion : Critério usado para medir a qualidade da divisão na árvore de decisão.
           - decision_tree_max_depth : Profundidade máxima da árvore de decisão.
           - decision_tree_max_feature : Número máximo de características a serem consideradas.
           - decision_tree_max_leaf : Número máximo de folhas na árvore de decisão.
           - ada_boost_estimator : Estimador base para o AdaBoost.
           - ada_boost_number_estimators : Número maximo de estimadores para o AdaBoost.
           - ada_boost_learning_rate : Taxa de aprendizado do AdaBoost.
           - ada_boost_algorithm : Algoritmo usado pelo AdaBoost.
           - naive_bayes_priors : Probabilidades de Priors para Naive Bayes.
           - naive_bayes_variation_smoothing : Parâmetro de suavização para Naive Bayes.
           - quadratic_discriminant_analysis_priors : Probabilidades de priors para QDA.
           - quadratic_discriminant_analysis_regularization : Parâmetro de regularização para QDA.
           - quadratic_discriminant_threshold : Limiar de decisão para QDA.
           - perceptron_training_algorithm : Algoritmo de treinamento para Perceptron.
           - perceptron_training_loss : Função de perda para o Perceptron.
           - perceptron_training_metric : Métrica de avaliação para o Perceptron.
           - perceptron_layer_activation : Função de ativação das camadas internas do Perceptron.
           - perceptron_last_layer_activation : Função de ativação da última camada do Perceptron.
           - perceptron_dropout_decay_rate : Taxa de decaimento para o dropout no Perceptron.
           - perceptron_number_epochs : Número de épocas de treinamento para o Perceptron.
           - perceptron_layers_settings : Configurações das camadas do Perceptron.
           - set_naive_bayes_priors : Define  a probabilidades de prior para QDA.
           - set_quadratic_discriminant_threshold : Define o limiar de decisão para QDA.

        """
        # Inicialização dos parâmetros privados com os valores fornecidos
        self.__random_forest_number_estimators = random_forest_number_estimators
        self.__random_forest_max_depth = random_forest_max_depth
        self.__max_leaf_nodes = max_leaf_nodes

        self.__knn_number_neighbors = knn_number_neighbors
        self.__knn_weights = knn_weights
        self.__knn_leaf_size = knn_leaf_size
        self.__knn_metric = knn_metric
        self.__knn_algorithm = knn_algorithm

        self.__support_vector_machine_normalization = support_vector_machine_normalization
        self.__support_vector_machine_kernel = support_vector_machine_kernel
        self.__support_vector_machine_kernel_degree = support_vector_machine_kernel_degree
        self.__support_vector_machine_gamma = support_vector_machine_gamma

        self.__gaussian_process_kernel = gaussian_process_kernel
        self.__gaussian_process_max_iterations = gaussian_process_max_iterations
        self.__gaussian_process_optimizer = gaussian_process_optimizer

        self.__decision_tree_criterion = decision_tree_criterion
        self.__decision_tree_max_depth = decision_tree_max_depth
        self.__decision_tree_max_feature = decision_tree_max_feature
        self.__decision_tree_max_leaf = decision_tree_max_leaf

        self.__ada_boost_estimator = ada_boost_estimator
        self.__ada_boost_number_estimators = ada_boost_number_estimators
        self.__ada_boost_learning_rate = ada_boost_learning_rate
        self.__ada_boost_algorithm = ada_boost_algorithm

        self.__naive_bayes_priors = naive_bayes_priors
        self.__naive_bayes_variation_smoothing = naive_bayes_variation_smoothing

        self.__quadratic_discriminant_analysis_priors = quadratic_discriminant_analysis_priors
        self.__quadratic_discriminant_regularize = quadratic_discriminant_analysis_regularization
        self.__quadratic_discriminant_threshold = quadratic_discriminant_threshold

        self.__perceptron_training_algorithm = perceptron_training_algorithm
        self.__perceptron_training_loss = perceptron_training_loss
        self.__perceptron_layer_activation = perceptron_layer_activation
        self.__perceptron_last_layer_activation = perceptron_last_layer_activation
        self.__perceptron_dropout_decay_rate = perceptron_dropout_decay_rate
        self.__perceptron_number_epochs = perceptron_number_epochs

        if perceptron_training_metric is None:
            self.__perceptron_training_metric = DEFAULT_PERCEPTRON_METRIC
        if perceptron_layers_settings is None:
            self.__perceptron_layers_settings = DEFAULT_PERCEPTRON_LAYERS_SETTINGS

    def __get_instance_random_forest(self, x_samples_training, y_samples_training, dataset_type):
        """
        Retorna uma instância treinada de um classificador RandomForest.
        
        Parâmetros:
           - x_samples_training : Amostras de treinamento.
           - y_samples_training : Rótulos das amostras de treinamento.
           - dataset_type : Tipo de dados das amostras.

        Retorno:
           - instance_model_classifier: instância treinada de um classificador RandomForest.
        """
        logging.info("    Starting training classifier: RANDOM FOREST")

        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)

        instance_model_classifier = RandomForestClassifier(n_estimators=self.__random_forest_number_estimators,
                                                           max_depth=self.__random_forest_max_depth,
                                                           max_leaf_nodes=self.__max_leaf_nodes)
        instance_model_classifier.fit(x_samples_training, y_samples_training)
        logging.info("\r    Finished training\n")

        return instance_model_classifier

    def __get_instance_k_neighbors_classifier(self, x_samples_training, y_samples_training, dataset_type):
        """
        Retorna uma instância treinada de um classificador KNN.
        
        Parâmetros:
          -  x_samples_training : Amostras de treinamento.
          -  y_samples_training : Rótulos das amostras de treinamento.
          -  dataset_type : Tipo de dados das amostras.

        Retorno:
          -  instance_model_classifier: instância treinada de um classificador KNN.
        """
        logging.info("    Starting training classifier: K-NEIGHBORS NEAREST")

        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)

        instance_model_classifier = KNeighborsClassifier(n_neighbors=self.__knn_number_neighbors,
                                                         weights=self.__knn_weights,
                                                         algorithm=self.__knn_algorithm,
                                                         leaf_size=self.__knn_leaf_size,
                                                         metric=self.__knn_metric)
        instance_model_classifier.fit(x_samples_training, y_samples_training)
        logging.info("\r    Finished training\n")

        return instance_model_classifier

    def __get_instance_support_vector_machine(self, x_samples_training, y_samples_training, dataset_type):

        """
        Retorna uma instância treinada de um classifcador SVM.
        
        Parâmetros:
           - x_samples_training : Amostras de treinamento.
           - y_samples_training : Rótulos das amostras de treinamento.
           - dataset_type : Tipo de dados das amostras.

        Retorno
           - instance_model_classifier: Instância treinada de um classifcador SVM.
        """
        logging.info("    Starting training classifier: SUPPORT VECTOR MACHINE")

        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)

        instance_model_classifier = SVC(C=self.__support_vector_machine_normalization,
                                        kernel=self.__support_vector_machine_kernel,
                                        degree=self.__support_vector_machine_kernel_degree,
                                        gamma=self.__support_vector_machine_gamma,probability=True)

        instance_model_classifier.fit(x_samples_training, y_samples_training)
        logging.info("\r    Finished training\n")

        return instance_model_classifier

    def __get_instance_gaussian_process(self, x_samples_training, y_samples_training, dataset_type):
        """
        Retorna uma instância treinada de um classificador de processo Gausiano.
        
        Parâmetros:
          -  x_samples_training : Amostras de treinamento.
          -  y_samples_training : Rótulos das amostras de treinamento.
          -  dataset_type : Tipo de dados das amostras.

        Retorno
           - instance_model_classifier Instância treinada de um classificador de processo Gausiano.
        """
        logging.info("    Starting training classifier: GAUSSIAN PROCESS")

        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)

        instance_model_classifier = GaussianProcessClassifier(kernel=self.__gaussian_process_kernel,
                                                              optimizer=self.__gaussian_process_optimizer,
                                                              max_iter_predict=self.__gaussian_process_max_iterations)

        instance_model_classifier.fit(x_samples_training, y_samples_training)
        logging.info("\r    Finished training\n")

        return instance_model_classifier
    def __get_instance_xgboost(self, x_samples_training, y_samples_training, dataset_type):
        """
        Retorna uma instância treinada de um classificador XGBoost.
        
        Parâmetros:
           - x_samples_training : Amostras de treinamento.
           - y_samples_training : Rótulos das amostras de treinamento.
           - dataset_type : Tipo de dados das amostras.

        Retorno
           - instance_model_classifier: Instância de um classificador XGBoost.
        """
        logging.info("    Starting training classifier: GAUSSIAN PROCESS")

        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)
        model = xgb.XGBClassifier()
        instance_model_classifier=model.fit(x_samples_training, y_samples_training)


        logging.info("\r    Finished training\n")

        return instance_model_classifier

    def __get_instance_SGDRegressor(self,x_samples_training, y_samples_training, dataset_type):
        """
        Retorna uma instância treinada de um classificador SGD.
        
        Parâmetros:
           - x_samples_training : Amostras de treinamento.
           -  y_samples_training : Rótulos das amostras de treinamento.
           -  dataset_type : Tipo de dados das amostras.

        Retorno
            instance_model_classifier: Instância de um classificador SGD.
        """
        logging.info("    Starting training classifier: SDG")
        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)
        clr = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000, tol=1e-3))
        clr.fit(x_samples_training, y_samples_training)
        calibrator = CalibratedClassifierCV(clr, cv='prefit')
        instance_model_classifier=calibrator.fit(x_samples_training,y_samples_training)

        logging.info("\r    Finished training\n")
        return instance_model_classifier
        
    
    
    def __get_instance_decision_tree(self, x_samples_training, y_samples_training, dataset_type):
        """
        Retorna uma instância treinada de um classificador Decision Tree.
        
        Parâmetros:
           - x_samples_training : Amostras de treinamento.
           -  y_samples_training : Rótulos das amostras de treinamento.
           -  dataset_type : Tipo de dados das amostras.

        Retorno:
            instance_model_classifier: Instância de um  classificador Decision Tree.
        """
        logging.info("    Starting training classifier: DECISION TREE")

        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)

        instance_model_classifier = DecisionTreeClassifier(criterion=self.__decision_tree_criterion,
                                                           max_depth=self.__decision_tree_max_depth,
                                                           max_features=self.__decision_tree_max_feature,
                                                           max_leaf_nodes=self.__decision_tree_max_leaf)

        instance_model_classifier.fit(x_samples_training, y_samples_training)
        logging.info("\r    Finished training\n")

        return instance_model_classifier

    def __get_instance_ada_boost(self, x_samples_training, y_samples_training, dataset_type):
        """
        Retorna uma instância treinada de um classificador  AdaBoost.
        
        Parâmetros:
           - x_samples_training : Amostras de treinamento.
           - y_samples_training : Rótulos das amostras de treinamento.
           - dataset_type : Tipo de dados das amostras.

        Retorno:
           - instance_model_classifier: Instância treinada de um classificador AdaBoost.
        """

        logging.info("    Starting training classifier: ADA BOOST")

        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)

        instance_model_classifier = AdaBoostClassifier(algorithm=self.__ada_boost_algorithm,
                                                       n_estimators=self.__ada_boost_number_estimators,
                                                       learning_rate=self.__ada_boost_learning_rate)

        instance_model_classifier.fit(x_samples_training, y_samples_training)
        logging.info("\r    Finished training\n")

        return instance_model_classifier

    def __get_instance_naive_bayes(self, x_samples_training, y_samples_training, dataset_type):
        """
        Retorna uma instância treinada de um classificador  Naive Bayes.
        
        Parâmetros:
           - x_samples_training : Amostras de treinamento.
           - y_samples_training : Rótulos das amostras de treinamento.
           - dataset_type : Tipo de dados das amostras.

        Retorno:
           - instance_model_classifier: Instância treinada de um classificador ANaive Bayes.
        """

        logging.info("    Starting training classifier: NAIVE BAYES")

        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)

        instance_model_classifier = GaussianNB(priors=self.__naive_bayes_priors,
                                               var_smoothing=self.__naive_bayes_variation_smoothing)

        instance_model_classifier.fit(x_samples_training, y_samples_training)
        logging.info("\r    Finished training\n")

        return instance_model_classifier

    def __get_instance_quadratic_discriminant(self, x_samples_training, y_samples_training, dataset_type):
        """
        Retorna uma instância treinada de um classificador discriminador quadrático.

        
        Parâmetros:
           - x_samples_training : Amostras de treinamento.
           - y_samples_training : Rótulos das amostras de treinamento.
           - dataset_type : Tipo de dados das amostras.

        Retorno:
           - instance_model_classifier: Instância treinada de um classificador  discriminador quadrático.
        """

        logging.info("    Starting training classifier: QUADRATIC DISCRIMINANT ANALYSIS")

        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)
        instance_model_classifier = QuadraticDiscriminantAnalysis(priors=self.__quadratic_discriminant_analysis_priors,
                                                                  reg_param=self.__quadratic_discriminant_regularize,
                                                                  tol=self.__quadratic_discriminant_threshold)
        instance_model_classifier.fit(x_samples_training, y_samples_training)
        logging.info("\r    Finished training\n")

        return instance_model_classifier

    def __get_instance_perceptron(self, x_samples_training, y_samples_training, dataset_type, verbose_level,
                                  input_dataset_shape):
        """
        Retorna uma instância treinada de um classificador perceptron de multi camadas.

        
        Parâmetros:
           - x_samples_training : Amostras de treinamento.
           - y_samples_training : Rótulos das amostras de treinamento.
           - dataset_type : Tipo de dados das amostras.

        Retorno:
           - instance_model_classifier: Instância treinada de um classificador  perceptron de multi camadas.
        """

        logging.info("    Starting training classifier: MULTILAYER PERCEPTRON")

        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)
        instance_model_classifier = PerceptronMultilayer(self.__perceptron_layers_settings,
                                                         self.__perceptron_training_metric,
                                                         self.__perceptron_training_loss,
                                                         self.__perceptron_training_algorithm,
                                                         dataset_type, self.__perceptron_layer_activation,
                                                         self.__perceptron_last_layer_activation,
                                                         self.__perceptron_dropout_decay_rate)
        model_classifier = instance_model_classifier.get_model(input_dataset_shape)
        model_classifier.fit(x_samples_training, y_samples_training, epochs=self.__perceptron_number_epochs,
                             verbose=DEFAULT_VERBOSE_LIST[verbose_level])
        logging.info("\r    Finished training\n")

        #calibrated_clf = CalibratedClassifierCV(model_classifier, cv="prefit")
        #calibrated_clf.fit(x_samples_training, y_samples_training)
        #return calibrated_clf
        return model_classifier

    def get_trained_classifiers(self, classifiers_list, x_samples_training, y_samples_training, dataset_type,
                                verbose_level, input_dataset_shape):
        """
        Retorna uma lista de classificadores treinados baseados na lista de classificadores fornecida.
        
        Parâmetros:
           - classifier_list : Lista de strings com nomes de algoritmos de classificação desejados.
           - x_samples_training : Amostras de treinamento.
           - y_samples_training: Rótulos das amostras de treinamento.
           - dataset_type : Tipo de dados das amostras.
           - log_level : Nível de log desejado para registrar o progresso.

        Retorno:
           - list_instance_classifiers: Lista contendo instâncias treinadas dos classificadores especificados.
        """
        logging.info("\nStarting training classifier\n")
        list_instance_classifiers = []

        for classifier_algorithm in classifiers_list:

            if classifier_algorithm == DEFAULT_CLASSIFIER_LIST[0]:

                list_instance_classifiers.append(self.__get_instance_random_forest(x_samples_training,
                                                                                   y_samples_training,
                                                                                   dataset_type))

            elif classifier_algorithm == DEFAULT_CLASSIFIER_LIST[1]:

                list_instance_classifiers.append(self.__get_instance_support_vector_machine(x_samples_training,
                                                                                            y_samples_training,
                                                                                            dataset_type))

            elif classifier_algorithm == DEFAULT_CLASSIFIER_LIST[2]:

                list_instance_classifiers.append(self.__get_instance_k_neighbors_classifier(x_samples_training,
                                                                                            y_samples_training,
                                                                                            dataset_type))

            elif classifier_algorithm == DEFAULT_CLASSIFIER_LIST[3]:
                list_instance_classifiers.append(self.__get_instance_gaussian_process(x_samples_training,
                                                                                      y_samples_training,
                                                                                      dataset_type))

            elif classifier_algorithm == DEFAULT_CLASSIFIER_LIST[4]:

                list_instance_classifiers.append(self.__get_instance_decision_tree(x_samples_training,
                                                                                   y_samples_training,
                                                                                   dataset_type))

            elif classifier_algorithm == DEFAULT_CLASSIFIER_LIST[5]:

                list_instance_classifiers.append(self.__get_instance_ada_boost(x_samples_training,
                                                                               y_samples_training,
                                                                               dataset_type))

            elif classifier_algorithm == DEFAULT_CLASSIFIER_LIST[6]:

                list_instance_classifiers.append(self.__get_instance_naive_bayes(x_samples_training,
                                                                                 y_samples_training,
                                                                                 dataset_type))

            elif classifier_algorithm == DEFAULT_CLASSIFIER_LIST[7]:

                list_instance_classifiers.append(self.__get_instance_quadratic_discriminant(x_samples_training,
                                                                                            y_samples_training,
                                                                                            dataset_type))

            elif classifier_algorithm == DEFAULT_CLASSIFIER_LIST[8]:

                list_instance_classifiers.append(self.__get_instance_perceptron(x_samples_training,
                                                                                y_samples_training,
                                                                                dataset_type,
                                                                                verbose_level,
                                                                                input_dataset_shape))
            elif classifier_algorithm == DEFAULT_CLASSIFIER_LIST[9]:

                list_instance_classifiers.append(self.__get_instance_SGDRegressor(x_samples_training,
                                                                                y_samples_training,
                                                                                dataset_type))
            elif classifier_algorithm == DEFAULT_CLASSIFIER_LIST[10]:

                list_instance_classifiers.append(self.__get_instance_xgboost(x_samples_training,
                                                                                y_samples_training,
                                                                                dataset_type))
                                                                                

        return list_instance_classifiers

    # funções abaixo definem os parâmetros de entrada especificados para o instanciamento de cada classificador
    def set_random_forest_number_estimators(self, random_forest_number_estimators):
        """
        Define o número de estimadores para o Random Forest.

        Parâmetros:
           - random_forest_number_estimators : Número de estimadores para o Random Forest.
        """
        self.__random_forest_number_estimators = random_forest_number_estimators

    def set_random_forest_max_depth(self, random_forest_max_depth):
        """
        Define a profundidade máxima da árvore do Random Forest.

        Parâmetros:
           - random_forest_max_depth : Profundidade máxima da árvore do Random Forest.
        """
        self.__random_forest_max_depth = random_forest_max_depth

    def max_leaf_nodes(self, max_leaf_nodes):
        """
        Define o número máximo de folhas no Random Forest.

        Parâmetros:
           - max_leaf_nodes: Número máximo de folhas no Random Forest.
        """
        self.__max_leaf_nodes = max_leaf_nodes

    def set_knn_number_neighbors(self, knn_number_neighbors):
        """
        Define o número de vizinhos para o classificador KNN.

        Parâmetros:
           - knn_number_neighbors: Número de vizinhos para o classificador KNN.
        """
        self.__knn_number_neighbors = knn_number_neighbors

    def set_knn_weights(self, knn_weights):
        """
        Define a função de peso para o classificador KNN.

        Parâmetros:
           - knn_weights: Função de peso para o classificador KNN.
        """
        self.__knn_weights = knn_weights

    def set_knn_leaf_size(self, knn_leaf_size):
        """
        Define o tamanho da folha para o KNN.

        Parâmetros:
           - knn_leaf_size: Tamanho da folha para o KNN
        """

        self.__knn_leaf_size = knn_leaf_size

    def set_knn_metric(self, knn_metric):
        """
        Define a  métrica utilizada para a computação da distância para o KNN.

        Parâmetros:
           - knn_metric: Métrica utilizada para a computação da distância para o KNN.
        """
        self.__knn_metric = knn_metric

    def set_knn_algorithm(self, knn_algorithm):
        """
        Define o algoritmo usado pelo KNN.

        Parâmetros:
           - knn_algorithm: Algoritmo usado pelo KNN.
        """
        self.__knn_algorithm = knn_algorithm

    def set_support_vector_machine_normalization(self, support_vector_machine_normalization):
        """
        Define o parâmetro de regularização para SVM.

        Parâmetros:
           - support_vector_machine_normalization: Parâmetro de regularização para SVM.
        """       
        self.__support_vector_machine_normalization = support_vector_machine_normalization

    def set_support_vector_machine_kernel(self, support_vector_machine_kernel):
        """
        Define o kernel usado pelo SVM.

        Parâmetros:
           - support_vector_machine_kernel:  Kernel usado pelo SVM.
        """       
        self.__support_vector_machine_kernel = support_vector_machine_kernel

    def set_support_vector_machine_kernel_degree(self, support_vector_machine_kernel_degree):
        """
        Define o grau do kernel para SVM.

        Parâmetros:
           - support_vector_machine_kernel_degree:  Grau do kernel para SVM.
        """       
        
        self.__support_vector_machine_kernel_degree = support_vector_machine_kernel_degree

    def set_support_vector_machine_gamma(self, support_vector_machine_gamma):
        """
        Define o coeficiente gamma para SVM.

        Parâmetros:
           - support_vector_machine_gamma: Coeficiente gamma para SVM.
        """       
        
        self.__support_vector_machine_gamma = support_vector_machine_gamma

    def set_gaussian_process_kernel(self, gaussian_process_kernel):
        """
        Define  o kernel para o processo gaussiano.

        Parâmetros:
           - gaussian_process_kernel: Kernel para o processo gaussiano.
        """              
        self.__gaussian_process_kernel = gaussian_process_kernel

    def set_gaussian_process_max_iterations(self, gaussian_process_max_iterations):
        """
        Define  o número máximo de iterações para o processo gaussiano.

        Parâmetros:
           - gaussian_process_max_iterations: Número máximo de iterações para o processo gaussiano.
        """                     
        self.__gaussian_process_max_iterations = gaussian_process_max_iterations

    def set_gaussian_process_optimizer(self, gaussian_process_optimizer):
        """
        Define  o otimizador para o processo gaussiano.

        Parâmetros:
           - gaussian_process_optimizer: Otimizador para o processo gaussiano.
        """    
        self.__gaussian_process_optimizer = gaussian_process_optimizer

    def set_decision_tree_criterion(self, decision_tree_criterion):
        """
        Define o critério usado para medir a qualidade da divisão na árvore de decisão.

        Parâmetros:
           - decision_tree_criterion: Critério usado para medir a qualidade da divisão na árvore de decisão.
        """    
        self.__decision_tree_criterion = decision_tree_criterion

    def set_decision_tree_max_depth(self, decision_tree_max_depth):
        """
        Define a profundidade máxima da árvore de decisão.

        Parâmetros:
           - decision_tree_max_depth: Profundidade máxima da árvore de decisão.
        """    
        self.__decision_tree_max_depth = decision_tree_max_depth

    def set_decision_tree_max_feature(self, decision_tree_max_feature):
        """
        Define o número máximo de características a serem consideradas para a árvore de decisão.

        Parâmetros:
           - decision_tree_max_feature: Número máximo de características a serem consideradas para a árvore de decisão..
        """    
        self.__decision_tree_max_feature = decision_tree_max_feature

    def set_decision_tree_max_leaf(self, decision_tree_max_leaf):
        """
        Define o número máximo de folhas na árvore de decisão.

        Parâmetros:
           - decision_tree_max_leaf: Número máximo de folhas na árvore de decisão.
        """    
        self.__decision_tree_max_leaf = decision_tree_max_leaf

    def set_ada_boost_estimator(self, ada_boost_estimator):
        """
        Define o estimador base para AdaBoost.

        Parâmetros:
           - ada_boost_estimator: Estimador base para AdaBoost.
        """           
        self.__ada_boost_estimator = ada_boost_estimator

    def set_ada_boost_number_estimators(self, ada_boost_number_estimators):
        """
        Define o  número maximo de estimadores para o AdaBoost.

        Parâmetros:
           - ada_boost_estimator: Número maximo de estimadores para o AdaBoost.
        """           
        self.__ada_boost_number_estimators = ada_boost_number_estimators

    def set_ada_boost_learning_rate(self, ada_boost_learning_rate):
        """
        Define a taxa de aprendizado do AdaBoost.

        Parâmetros:
           - ada_boost_learning_rate: Taxa de aprendizado do AdaBoost.
        """           
        self.__ada_boost_learning_rate = ada_boost_learning_rate

    def set_ada_boost_algorithm(self, ada_boost_algorithm):
        """
        Define o algoritmo a utilizado pelo AdaBoost.

        Parâmetros:
           - ada_boost_learning_rate: Algoritmo a utilizado pelo AdaBoost.
        """           
        self.__ada_boost_algorithm = ada_boost_algorithm

    def set_naive_bayes_priors(self, naive_bayes_priors):
        """
        Define a probabilidades de Priors para Naive Bayes.

        Parâmetros:
           - naive_bayes_priors: Probabilidades de Priors para Naive Bayes.
        """           
        self.__naive_bayes_priors = naive_bayes_priors

    def set_naive_bayes_variation_smoothing(self, naive_bayes_variation_smoothing):
        """
        Define o parâmetro de suavização para Naive Bayes.

        Parâmetros:
           - naive_bayes_variation_smoothing: Parâmetro de suavização para Naive Bayes.
        """           
        self.__naive_bayes_variation_smoothing = naive_bayes_variation_smoothing

    def set_quadratic_discriminant_analysis_priors(self, quadratic_discriminant_analysis_priors):
        """
        Define  a probabilidades de prios para QDA.

        Parâmetros:
           - quadratic_discriminant_analysis_priors: Probabilidades de priors para QDA.
        """           
        self.__quadratic_discriminant_analysis_priors = quadratic_discriminant_analysis_priors

    def set_quadratic_discriminant_analysis_regularization(self, quadratic_discriminant_analysis_regularization):
        """
        Define o parâmetro de regularização para QDA.

        Parâmetros:
           - quadratic_discriminant_analysis_regularization: Parâmetro de regularização para QDA.
        """    
        self.__quadratic_discriminant_regularize = quadratic_discriminant_analysis_regularization

    def set_quadratic_discriminant_threshold(self, quadratic_discriminant_threshold):
        """
        Define o limiar de decisão para QDA.

        Parâmetros:
           - quadratic_discriminant_threshold: Limiar de decisão para QDA.
        """    
        self.__quadratic_discriminant_threshold = quadratic_discriminant_threshold

    def set_perceptron_training_algorithm(self, perceptron_training_algorithm):
        """
        Define o algoritmo de treinamento para o Perceptron.

        Parâmetros:
           - perceptron_training_algorithm: Algoritmo de treinamento para o Perceptron.
        """    
        self.__perceptron_training_algorithm = perceptron_training_algorithm

    def set_perceptron_training_loss(self, perceptron_training_loss):
        """
        Define  a função de perda para o Perceptron.

        Parâmetros:
           - perceptron_training_loss: Função de perda para o Perceptron
        """    
        self.__perceptron_training_loss = perceptron_training_loss

    def set_perceptron_layer_activation(self, perceptron_layer_activation):
        """
        Define a função de ativação das camadas internas do Perceptron.

        Parâmetros:
           - perceptron_layer_activation: Função de ativação das camadas internas do Perceptron.
        """    
        self.__perceptron_layer_activation = perceptron_layer_activation

