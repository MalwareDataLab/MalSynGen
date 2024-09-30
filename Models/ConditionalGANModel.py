"""
Este módulo define a classe ConditionalGAN  responsavel por configurar e criar um modelo GAN condicional (cGAN) utilizando Keras.
Classes: 
    - Classe responsavel por configurar e criar um modelo GAN condicional utilizando Keras. 
"""
## Importação de bibliotecas necessárias
import numpy as np
from keras.layers import Input, Dense, Flatten, Dropout, Activation, LeakyReLU, PReLU, Concatenate, BatchNormalization
from keras.initializers import RandomNormal
from keras.models import Model

## Parâmetros padrão para a Conditional GAN
##Valor padrão para o valor da minesão latente
DEFAULT_CONDITIONAL_GAN_LATENT_DIMENSION = 128
##Algoritmo padrão de treinamento para a cGAN
DEFAULT_CONDITIONAL_GAN_TRAINING_ALGORITHM = "Adam"
##Função de ativação padrão para a cGAN
DEFAULT_CONDITIONAL_GAN_ACTIVATION = "LeakyReLU"
##Valor padrão para a taxa de dropout para o gerador
DEFAULT_CONDITIONAL_GAN_DROPOUT_DECAY_RATE_G = 0.2
##Valor padrão para a taxa de dropout para o discriminador
DEFAULT_CONDITIONAL_GAN_DROPOUT_DECAY_RATE_D = 0.4
##Valor padrão para o tamanho de lote
DEFAULT_CONDITIONAL_GAN_BATCH_SIZE = 32
##Valor padrão para o número de classes
DEFAULT_CONDITIONAL_GAN_NUMBER_CLASSES = 2
##Valor padrão para o tamanho das camadas densas do gerador.
DEFAULT_CONDITIONAL_GAN_DENSE_LAYERS_SETTINGS_G = [128]
##Valor padrão para o tamanho das camadas densas do discriminador.
DEFAULT_CONDITIONAL_GAN_DENSE_LAYERS_SETTINGS_D = [128]
##Função padrão de perda da GAN
DEFAULT_CONDITIONAL_GAN_LOSS = "binary_crossentropy"
##Valor padrão para o momentum da GAN
DEFAULT_CONDITIONAL_GAN_MOMENTUM = 0.8
##Função de ativação padrão para a ultima camada
DEFAULT_CONDITIONAL_LAST_ACTIVATION_LAYER = "sigmoid"
##Valor padrão para a inicialização dos pesos das camadas
DEFAULT_CONDITIONAL_GAN_INITIALIZER_MEAN = 0.0
##Valor padrão para  o desvio padrão para inicialização dos pesos das camadas
DEFAULT_CONDITIONAL_GAN_INITIALIZER_DEVIATION = 0.02


class ConditionalGAN:
    """
    Classe responsavel por configurar e criar um modelo GAN condicional (cGAN) utilizando Keras.
    Funções:
        - __init__ : Inicializa a ConditionalGAN com os parâmetros especificados ou padrão.
        - add_activation_layer :  Define a funlção de ativação especificada à  camada na rede neural.
        - get_generator : Instância e retorna o modelo gerador da ConditionalGAN.
        - get_discriminator : Instância e retorna o modelo gerador da ConditionalGAN.
        - get_dense_generator_model : Retorna o modelo do gerador da ConditionalGAN.
        - get_dense_discriminator_model :  Retorna o modelo do discriminador da ConditionalGAN.
        - set_latent_dimension : Define a dimensão latente da ConditionalGAN.
        - set_output_shape : Define o formato de saída da ConditionalGAN.
        - set_activation_function : Define a função de ativação das camadas internas da ConditionalGAN.
        - set_last_layer_activation :Define a função de ativação para a última camada do gerador.
        - set_dropout_decay_rate_generator : Define a taxa do dropout para o gerador da ConditionalGAN.
        - set_dropout_decay_rate_discriminator : Define a taxa do dropout para o discriminador da ConditionalGAN.
        - set_dense_layer_sizes_generator : Define os tamanhos das camadas densas para o gerador da ConditionalGAN.
        - set_dense_layer_sizes_discriminator : Define os tamanhos das camadas densas para o discriminador da ConditionalGAN.
        - set_dataset_type : Define o tipo de dados do dataset.
        - set_initializer_mean : Define a média para a inicialização dos pesos das camadas.
        - set_initializer_deviation : Define o desvio padrão para inicialização dos pesos das camadas.
    """

    def __init__(self, latent_dim=DEFAULT_CONDITIONAL_GAN_LATENT_DIMENSION, output_shape=None,
                 activation_function=DEFAULT_CONDITIONAL_GAN_ACTIVATION,
                 initializer_mean=DEFAULT_CONDITIONAL_GAN_INITIALIZER_MEAN,
                 initializer_deviation=DEFAULT_CONDITIONAL_GAN_INITIALIZER_DEVIATION,
                 dropout_decay_rate_g=DEFAULT_CONDITIONAL_GAN_DROPOUT_DECAY_RATE_G,
                 dropout_decay_rate_d=DEFAULT_CONDITIONAL_GAN_DROPOUT_DECAY_RATE_D,
                 last_layer_activation=DEFAULT_CONDITIONAL_LAST_ACTIVATION_LAYER,
                 dense_layer_sizes_g=None, dense_layer_sizes_d=None, dataset_type=np.float32):
        """
        Inicializa a ConditionalGAN com os parâmetros especificados ou padrão.

        Parâmetros:
           - latent_dim : Dimensão do espaço latente.
           - output_shape t: Formato da saída.
           - activation_function : Função de ativação para as camadas internas da cGAN.
           - initializer_mean : Média para inicialização dos pesos das camadas
           - initializer_deviation : Desvio padrão para inicialização dos pesos das camadas.
           - dropout_decay_rate_g : Taxa de dropout para o gerador.
           - dropout_decay_rate_d : Taxa de dropout para o discriminador.
           - last_layer_activation : Função de ativação para a última camada do gerador.
           - dense_layer_sizes_g : Tamanhos das camadas densas do gerador.
           - dense_layer_sizes_d : Tamanhos das camadas densas do discriminador.
           - dataset_type : Tipo de dados do dataset.
        """
        if dense_layer_sizes_d is None:
            dense_layer_sizes_d = DEFAULT_CONDITIONAL_GAN_DENSE_LAYERS_SETTINGS_D

        if dense_layer_sizes_g is None:
            dense_layer_sizes_g = DEFAULT_CONDITIONAL_GAN_DENSE_LAYERS_SETTINGS_G

        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.activation_function = activation_function
        self.last_layer_activation = last_layer_activation
        self.dropout_decay_rate_g = dropout_decay_rate_g
        self.dropout_decay_rate_d = dropout_decay_rate_d
        self.dense_layer_sizes_g = dense_layer_sizes_g
        self.dense_layer_sizes_d = dense_layer_sizes_d
        self.dataset_type = dataset_type
        self.initializer_mean = initializer_mean
        self.initializer_deviation = initializer_deviation
        self.generator_model_dense = None
        self.discriminator_model_dense = None

    def add_activation_layer(self, neural_nodel):
        """
        Define a funlção de ativação especificada à  camada na rede neural.

        Parâmetros:
           - neural_nodel : Camada da rede neural.

        Retorno:
           - neural_nodel: Camada de rede neural com função de ativação adicionada.
        """
        if self.activation_function == 'LeakyReLU':
            neural_nodel = LeakyReLU()(neural_nodel)

        elif self.activation_function == 'ReLU':
            neural_nodel = Activation('relu')(neural_nodel)

        elif self.activation_function == 'PReLU':
            neural_nodel = PReLU()(neural_nodel)

        return neural_nodel

    def get_generator(self):
        """
        Instância e retorna o modelo gerador da ConditionalGAN.

         Retorno:
           - Model: Modelo do gerador.
        """
        initialization = RandomNormal(mean=self.initializer_mean, stddev=self.initializer_deviation)
        neural_model_inputs = Input(shape=(self.latent_dim,), dtype=self.dataset_type)
        latent_input = Input(shape=(self.latent_dim,))

        label_input = Input(shape=(1,), dtype=self.dataset_type)

        generator_model = Dense(self.dense_layer_sizes_g[0], kernel_initializer=initialization)(neural_model_inputs)
        #generator_model = BatchNormalization(momentum=0.8)(generator_model)
        generator_model = Dropout(self.dropout_decay_rate_g)(generator_model)
        generator_model = self.add_activation_layer(generator_model)

        for layer_size in self.dense_layer_sizes_g[1:]:
            generator_model = Dense(layer_size, kernel_initializer=initialization)(generator_model)
            #generator_model = BatchNormalization(momentum=0.8)(generator_model)
            generator_model = Dropout(self.dropout_decay_rate_g)(generator_model)
            generator_model = self.add_activation_layer(generator_model)

        generator_model = Dense(self.output_shape,activation='sigmoid', kernel_initializer=initialization)(
            generator_model)
        generator_model = Model(neural_model_inputs, generator_model, name="Dense_Generator")
        self.generator_model_dense = generator_model
        concatenate_output = Concatenate()([latent_input, label_input])
        label_embedding = Flatten()(concatenate_output)
        model_input = Dense(self.latent_dim)(label_embedding)
        model_input = self.add_activation_layer(model_input)
        generator_output_flow = generator_model(model_input)
        return Model([latent_input, label_input], generator_output_flow, name="Generator")

    def get_discriminator(self):
        """
        Instância e retorna o modelo discriminador da ConditionalGAN.

        Retorno:
           - Model: Modelo do discriminador.
        """
        neural_model_input = Input(shape=(self.output_shape//16,), dtype=self.dataset_type)
        discriminator_shape_input = Input(shape=(self.output_shape,))
        label_input = Input(shape=(1,), dtype=self.dataset_type)

        discriminator_model = Dense(self.dense_layer_sizes_d[0]//2,name="pesado")(neural_model_input)
        discriminator_model = Dropout(self.dropout_decay_rate_d)(discriminator_model)
        #discriminator_model = BatchNormalization(momentum=0.8)(discriminator_model)
        discriminator_model = self.add_activation_layer(discriminator_model)

        for layer_size in self.dense_layer_sizes_d[1:]:

            discriminator_model = Dense(layer_size//2,)(discriminator_model)
            #discriminator_model = BatchNormalization(momentum=0.8)(discriminator_model)
            discriminator_model = Dropout(self.dropout_decay_rate_d)(discriminator_model)
            discriminator_model = self.add_activation_layer(discriminator_model)

        discriminator_model = Dense(1, self.last_layer_activation,name='primeiro_neuronio')(discriminator_model)
        discriminator_model = Model(inputs=neural_model_input, outputs=discriminator_model, name="segundo_neuronio")
        self.discriminator_model_dense = discriminator_model
        concatenate_output = Concatenate()([discriminator_shape_input, label_input])
        label_embedding = Flatten()(concatenate_output)
        model_input = Dense(self.output_shape//16)(label_embedding)
        model_input = self.add_activation_layer(model_input)
        validity = discriminator_model(model_input)
        return Model(inputs=[discriminator_shape_input, label_input], outputs=validity, name='Discriminator')

    def get_dense_generator_model(self):
        """
        Retorna o modelo do gerador da ConditionalGAN.

        Retorno:
           - self.generator_model_dense: Modelo do gerador.
        """
        return self.generator_model_dense

    def get_dense_discriminator_model(self):
        """
        Retorna o modelo do discriminador da ConditionalGAN.

        Retorno:
           - self.discriminator_model_dense: Modelo do discriminador.
        """
        return self.discriminator_model_dense

    def set_latent_dimension(self, latent_dimension):
        """
        Define a dimensão latente da ConditionalGAN.

        Parâmetros:
           - latent_dimension : Dimensão latente.
        """
        self.latent_dim = latent_dimension

    def set_output_shape(self, output_shape):
        """
        Define o formato de saída da ConditionalGAN.

        Parâmetros:
          -  output_shape : Formato de saída da ConditionalGAN.
        """
        self.output_shape = output_shape

    def set_activation_function(self, activation_function):
        """
        Define a função de ativação das camadas internas da ConditionalGAN.

        Parâmetros:
          -  activation_function: Função de ativação.
        """
        self.activation_function = activation_function

    def set_last_layer_activation(self, last_layer_activation):
        """
        Define a função de ativação para a última camada do gerador.

        Parâmetros:
          -  last_layer_activation: Função de ativação para a última camada do gerador.
        """
        self.last_layer_activation = last_layer_activation

    def set_dropout_decay_rate_generator(self, dropout_decay_rate_generator):
        """
        Define a taxa do dropout para o gerador da ConditionalGAN.

        Parâmetros:
          -  dropout_decay_rate_generator: Taxa  do dropout para o gerador.
        """
        self.dropout_decay_rate_g = dropout_decay_rate_generator

    def set_dropout_decay_rate_discriminator(self, dropout_decay_rate_discriminator):
        """
        Define a taxa do dropout para o discriminador da ConditionalGAN.

        Parâmetros:
          -  dropout_decay_rate_discriminator: Taxa do dropout para o discriminador.
        """
        self.dropout_decay_rate_d = dropout_decay_rate_discriminator

    def set_dense_layer_sizes_generator(self, dense_layer_sizes_generator):
        """
        Define os tamanhos das camadas densas para o gerador da ConditionalGAN.

        Parâmetros:
           - dense_layer_sizes_generator: Lista de tamanhos das camadas densas para o gerador.
        """
        self.dense_layer_sizes_g = dense_layer_sizes_generator

    def set_dense_layer_sizes_discriminator(self, dense_layer_sizes_discriminator):
        """
        Define os tamanhos das camadas densas para o discriminador da ConditionalGAN.

        Parâmetros:
           - dense_layer_sizes_discriminator : Lista de tamanhos das camadas densas para o discriminador.
        """
        self.dense_layer_sizes_d = dense_layer_sizes_discriminator

    def set_dataset_type(self, dataset_type):
        """
        Define o tipo de dados do dataset.

        Parâmetros:
          -  dataset_type : Tipo de dado do dataset.
        """
        self.dataset_type = dataset_type

    def set_initializer_mean(self, initializer_mean):
        """
        Define a média para a inicialização dos pesos das camadas.

        Parâmetros:
          -  initializer_mean : média para inicialização dos pesos das camadas.
        """
        self.initializer_mean = initializer_mean

    def set_initializer_deviation(self, initializer_deviation):
        """
        Define o desvio padrão para inicialização dos pesos das camadas.

        Parâmetros:
           - initializer_deviation : desvio padrão para inicialização dos pesos das camadas.
        """
        self.initializer_deviation = initializer_deviation
