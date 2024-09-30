"""
Este módulo define a classe PerceptronMutlilayer e suas funções utilizadas para a classificação de dados binarios
Classes:
    -PerceptronMultilayer (Responsável pelo instanciamento e criação de um perceptron de múltiplas camadas(MLP))
"""
# Importação de bibliotecas necessárias
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from tensorflow import keras
import numpy as np

## Algoritmo padrão de treinamento
DEFAULT_PERCEPTRON_TRAINING_ALGORITHM = "Adam"
## Função de perda padrão para o perceptron
DEFAULT_PERCEPTRON_LOSS = "binary_crossentropy"
## Configuração de camadas padrão para o Perceptron
DEFAULT_PERCEPTRON_LAYERS_SETTINGS = [512, 256, 256]
## Valor padrão para a taxa de dropouut
DEFAULT_PERCEPTRON_DROPOUT_DECAY_RATE = 0.2
## Métricas padrão utilizada para a avaliação do modelo.
DEFAULT_PERCEPTRON_METRIC = ["accuracy"]
## Função padrão de ativação para as camadas internas do modelo
DEFAULT_PERCEPTRON_LAYER_ACTIVATION = keras.activations.swish
## Função padrão de ativação para a última camada do modelo
DEFAULT_PERCEPTRON_LAST_LAYER_ACTIVATION = "sigmoid"
## Valor padrão de entrada para o perceptron
DEFAULT_PERCEPTRON_DATA_TYPE = np.float32


class PerceptronMultilayer:
    """
     A classe PerceptronMultilayer é responsável pelo instanciamento e criação de um perceptron de múltiplas camadas(MLP)
     Funções:
        - __init__ : Inicializa o PerceptronMultilayer com as configurações especificadas
        - get_model :  Compila a instância do modelo com bases nas configuraçõe especificadas.
        - set_training_algorithm : Defino o algoritmo de treinamento a ser utilizado para optimização durante o treinamento do modelo.
        - set_training_loss : Define a função de perda a ser utilizada.
        - set_data_type: Define o tipo de dados para a entrada do modelo.
        - set_last_layer_activation :  Define a função de ativação para a última camada do modelo.
        - set_dropout_decay_rate :  Define a taxa de dropout para o modelo

    Parâmetros:
        - training_algorithm : Algoritmo de treinamento a ser utilizado.
        - training_loss : A função de perda utilizada.
        - data_type :  O tipo de dados utilizada para a entrada do modelo.
        - layer_activation : Função de ativação das camadas ocultas.
        - last_layer_activation: Função de ativação para a última camada do modelo.
        - dropout_decay_rate : Taxa de dropout.
        - training_metric : Métricas utilizadas para a avaliação do modelo.
        - layers_settings : Número de neurônios  por camada.
    """

    def __init__(self, layers_settings=None, training_metric=None, training_loss=DEFAULT_PERCEPTRON_LOSS,
                 training_algorithm=DEFAULT_PERCEPTRON_TRAINING_ALGORITHM, data_type=DEFAULT_PERCEPTRON_DATA_TYPE,
                 layer_activation=DEFAULT_PERCEPTRON_LAYER_ACTIVATION,
                 last_layer_activation=DEFAULT_PERCEPTRON_LAST_LAYER_ACTIVATION,
                 dropout_decay_rate=DEFAULT_PERCEPTRON_DROPOUT_DECAY_RATE):
        """
        Inicializa o PerceptronMultilayer com as configurações especificadas.

        Parâmetros:
            - layers_settings : Número de neurônios  por camada.
            - training_metric : Métricas utilizadas para a avaliação do modelo.
            - training_loss : A função de perda utilizada
            - training_algorithm : : Algoritmo de treinamento a ser utilizado.
            - data_type : O tipo de dados utilizada para a entrada do modelo.
            - layer_activation : Função de ativação das camadas ocultas.
            - last_layer_activation : Função de ativação para a última camada do gerador .
            - dropout_decay_rate : Taxa de dropout.
        """
        self.training_algorithm = training_algorithm
        self.training_loss = training_loss
        self.data_type = data_type
        self.layer_activation = layer_activation
        self.last_layer_activation = last_layer_activation
        self.dropout_decay_rate = dropout_decay_rate

        if training_metric is None:
            training_metric = DEFAULT_PERCEPTRON_METRIC

        if layers_settings is None:
            layers_settings = DEFAULT_PERCEPTRON_LAYERS_SETTINGS

        self.training_metric = training_metric
        self.layers_settings = layers_settings

    def get_model(self, input_shape):
        """
       Compila a instância do modelo com bases nas configuraçõe especificadas.

        Parâmetros:
            input_shape: Formato da entrada.

        Retorna
            keras.Model: Modelo Keras compilado representando o Perceptron.
        """
        input_layer = Input(shape=input_shape, dtype=self.data_type)
        dense_layer = Dense(self.layers_settings[0], activation=self.layer_activation)(input_layer)

        for num_neurons in self.layers_settings[1:]:
            dense_layer = Dense(num_neurons, activation=self.layer_activation)(dense_layer)
            dense_layer = Dropout(self.dropout_decay_rate)(dense_layer)

        output_layer = Dense(1, activation=self.last_layer_activation)(dense_layer)

        perceptron_model = Model(input_layer, output_layer)
        perceptron_model.compile(optimizer=self.training_algorithm,
                                 loss=self.training_loss,
                                 metrics=self.training_metric)

        return perceptron_model

    def set_training_algorithm(self, training_algorithm):
        """
        Defino o algoritmo de treinamento a ser utilizado para optimização durante o treinamento do modelo.

        Parâmetros:
            training_algorithm : Algoritmo de treinamento a ser utilizado.
        """
        self.training_algorithm = training_algorithm

    def set_training_loss(self, training_loss):
        """
        Define a função de perda a ser utilizada.

        Parâmetros:
            training_loss: Função da perda a ser utilizada.
        """
        self.training_loss = training_loss

    def set_data_type(self, data_type):
        """
        Define o tipo de dados para a entrada do modelo.

        Parâmetros:
            data_type: Tipo de dados para a entrada do modelo.
        """
        self.data_type = data_type

    def set_layer_activation(self, layer_activation):
        """
        Define a função de ativação para as camadas internas do modelo.

        Parâmetros:
            layer_activation : Função de ativação das camadas internas do modelo.
        """
        self.layer_activation = layer_activation

    def set_last_layer_activation(self, last_layer_activation):
        """

        Define a função de ativação para a última camada do modelo.

        Parâmetros:
            last_layer_activation : Função de ativação para a última camada do modelo.
        """
        self.last_layer_activation = last_layer_activation

    def set_dropout_decay_rate(self, dropout_decay_rate):
        """
        Define a taxa de dropout para o modelo

        Parâmetros:

            dropout_decay_rate : Taxa de dropout.
        """
        self.dropout_decay_rate = dropout_decay_rate

