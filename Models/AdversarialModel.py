"""
Este módulo define a classe AdversarialModel, que é utilizada a rede adversarial generativa, carregar e salvar modelos, estabelecer otimizadores e os passos de treino de cada dobra.
Classes:
- AdversarialModel (Classe utilizada para a criação e treinamento um modelo adversarial generativo (GAN)).
"""

## Importação de bibliotecas necessárias
import logging
import os
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers.legacy import Adam, RMSprop, Adadelta
import tensorflow as tf

### Constantes padrão para hiperparâmetros do modelo adversarial

DEFAULT_OPTIMIZER_GENERATOR_LEARNING = 0.0001 ##Valor padrão para a taxa de aprendizado do gerador

DEFAULT_OPTIMIZER_DISCRIMINATOR_LEARNING = 0.0001 ##Valor padrão para a taxa de aprendizado do discriminador
## Valor padrão beta para o gerador
DEFAULT_OPTIMIZER_GENERATOR_BETA = 0.5
##Valor padrão beta para o  discriminador
DEFAULT_OPTIMIZER_DISCRIMINATOR_BETA = 0.5
DEFAULT_LATENT_DIMENSION = 128
## Otimizadores padrões para o gerador e discriminador
DEFAULT_OPTIMIZER_GENERATOR = Adam(DEFAULT_OPTIMIZER_GENERATOR_LEARNING, DEFAULT_OPTIMIZER_GENERATOR_BETA)
DEFAULT_OPTIMIZER_DISCRIMINATOR = Adam(DEFAULT_OPTIMIZER_DISCRIMINATOR_LEARNING, DEFAULT_OPTIMIZER_DISCRIMINATOR_BETA)
##Funções de perda padrão para o gerador 
DEFAULT_LOSS_GENERATOR = BinaryCrossentropy()
##Funções de perda padrão parao o discriminador
DEFAULT_LOSS_DISCRIMINATOR = BinaryCrossentropy()

##Valor padrão para a taxa de aprendizado para o otimizador Adam.
DEFAULT_CONDITIONAL_GAN_ADAM_LEARNING_RATE = 0.0001
##Valor padrão beta para o otimizador Adam.
DEFAULT_CONDITIONAL_GAN_ADAM_BETA = 0.5
##Valor padrão para a Taxa de aprendizado para o otimizador Adam.
DEFAULT_CONDITIONAL_GAN_RMS_PROP_LEARNING_RATE = 0.001
##Valor padrão para a taxa de decaimento para o otimizador RMSprop.
DEFAULT_CONDITIONAL_GAN_RMS_PROP_DECAY_RATE = 0.5
##Valor padrão para a taxa de aprendizado para o otimizador Adadelta.
DEFAULT_CONDITIONAL_GAN_ADA_DELTA_LEARNING_RATE = 0.001
##Valor padrão para a taxa de decaimento para o otimizador Adadelta.
DEFAULT_CONDITIONAL_GAN_ADA_DELTA_DECAY_RATE = 0.5
##Valor padrão para a taxa de suavização do modelo adversarial
DEFAULT_CONDITIONAL_GAN_SMOOTHING_RATE = 0.15
##Valor padrão para a média da distribuição latente do modelo adversarial.
DEFAULT_CONDITIONAL_GAN_LATENT_MEAN_DISTRIBUTION = 0.0
##Valor padrão para o desvio padrão da distribuição latente do modelo adversarial
DEFAULT_CONDITIONAL_GAN_LATENT_STANDER_DEVIATION = 1.0

##Títulos padrão para os arquivos dos modelos 
DEFAULT_FILE_NAME_DISCRIMINATOR = "discriminator_model"
DEFAULT_FILE_NAME_GENERATOR = "generator_model"
DEFAULT_PATH_OUTPUT_MODELS = "models_saved/"

class AdversarialModel(Model):
    """
    Classe utilizada para a criação e treinamento um modelo adversarial generativo .
    Funções:
        - __init__ : Inicializa a classe com valores padrão ou fornecidos.
        - get_learning_rates : Retorna as taxas de aprendizado dos otimizadores.
        - compile :  Compila o modelo adversarial com os otimizadores e funções de perda fornecidos.
        - train_step : Executa uma etapa de treino para o modelo adversarial
        - save_models : Salva os modelos do gerador e discriminador em arquivos JSON e H5.
        - load_models : Carrega os modelos gerador e discriminador de arquivos JSON e H5.
        - set_generator : Define o modelo gerador para  o modelo adversarial.
        - set_discriminator : Define o modelo discriminador para o modelo adversarial.
        - set_latent_dimension :  Define a dimensão do espaço latente.
        - set_optimizer_generator : Define o otimizador para o gerador.
        - set_optimizer_discriminator : Define o otimizador para o discriminador. 
        - set_loss_generator : Define a função de perda para o gerador.
        - set_loss_discriminator : Define a função de perda para o discriminador.
        - get_optimizer : Retorna o otimizador baseado no algoritmo de treinamento especificado.
    """
    def __init__(self, generator_model=None, discriminator_model=None, latent_dimension=DEFAULT_LATENT_DIMENSION,
                 optimizer_generator=DEFAULT_OPTIMIZER_GENERATOR, loss_generator=DEFAULT_LOSS_GENERATOR,
                 optimizer_discriminator=DEFAULT_OPTIMIZER_DISCRIMINATOR, loss_discriminator=DEFAULT_LOSS_DISCRIMINATOR,
                 conditional_gan_adam_learning_rate=DEFAULT_CONDITIONAL_GAN_ADAM_LEARNING_RATE,
                 conditional_gan_adam_beta=DEFAULT_CONDITIONAL_GAN_ADAM_BETA,
                 conditional_gan_rms_prop_learning_rate=DEFAULT_CONDITIONAL_GAN_RMS_PROP_LEARNING_RATE,
                 conditional_gan_rms_prop_decay_rate=DEFAULT_CONDITIONAL_GAN_RMS_PROP_DECAY_RATE,
                 conditional_gan_ada_delta_learning_rate=DEFAULT_CONDITIONAL_GAN_ADA_DELTA_LEARNING_RATE,
                 conditional_gan_ada_delta_decay_rate=DEFAULT_CONDITIONAL_GAN_ADA_DELTA_DECAY_RATE,
                 file_name_discriminator=DEFAULT_FILE_NAME_DISCRIMINATOR,
                 file_name_generator=DEFAULT_FILE_NAME_GENERATOR, models_saved_path=DEFAULT_PATH_OUTPUT_MODELS,
                 latent_mean_distribution=DEFAULT_CONDITIONAL_GAN_LATENT_MEAN_DISTRIBUTION,
                 latent_stander_deviation=DEFAULT_CONDITIONAL_GAN_LATENT_MEAN_DISTRIBUTION,
                 smoothing_rate=DEFAULT_CONDITIONAL_GAN_SMOOTHING_RATE, *args, **kwargs):
        """
        Inicializa a classe AdversarialModel com os parâmetros fornecidos.

        Parâmetros:
          -  generator_model: O modelo gerador.
          -  discriminator_model : O modelo discriminador.
          -  latent_dimension : Dimensão do espaço latente.
          -  optimizer_generator : Otimizador para o gerador.
          -  loss_generator : Função de perda para o gerador.
          -  optimizer_discriminator : Otimizador para o discriminador.
          -  loss_discriminator : Função de perda para o discriminador.
          -  conditional_gan_adam_learning_rate : Taxa de aprendizado para o otimizador Adam.
          -  conditional_gan_adam_beta : Valor beta para o otimizador Adam.
          -  conditional_gan_rms_prop_learning_rate : Taxa de aprendizado para o otimizador RMSprop.
          -  conditional_gan_rms_prop_decay_rate : Taxa de decaimento para o otimizador RMSprop.
          -  conditional_gan_ada_delta_learning_rate : Taxa de aprendizado para o otimizador Adadelta.
          -  conditional_gan_ada_delta_decay_rate : Taxa de decaimento para o otimizador Adadelta.
          -  file_name_discriminator : Nome do arquivo para salvar o modelo discriminador.
          -  file_name_generator : Nome do arquivo para salvar o modelo gerador.
          -  models_saved_path : Caminho para salvar os modelos.
          -  latent_mean_distribution : Média da distribuição latente do modelo adversarial.
          -  latent_stander_deviation : Desvio padrão da distribuição latente do modelo adversarial.
          -  smoothing_rate : Taxa de suavização do modelo adversarial.
        """
        super().__init__(*args, **kwargs)

        self.generator = generator_model
        self.discriminator = discriminator_model
        self.latent_dimension = latent_dimension
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator
        self.loss_generator = loss_generator
        self.loss_discriminator = loss_discriminator

        self.conditional_gan_adam_learning_rate = conditional_gan_adam_learning_rate
        self.conditional_gan_adam_beta = conditional_gan_adam_beta
        self.conditional_gan_rms_prop_learning_rate = conditional_gan_rms_prop_learning_rate
        self.conditional_gan_rms_prop_decay_rate = conditional_gan_rms_prop_decay_rate
        self.conditional_gan_ada_delta_learning_rate = conditional_gan_ada_delta_learning_rate
        self.conditional_gan_ada_delta_decay_rate = conditional_gan_ada_delta_decay_rate

        self.smoothing_rate = smoothing_rate
        self.latent_mean_distribution = latent_mean_distribution
        self.latent_stander_deviation = latent_stander_deviation

        self.file_name_discriminator = file_name_discriminator
        self.file_name_generator = file_name_generator
        self.models_saved_path = models_saved_path

    def get_learning_rates(self):
        """
        Retorna as taxas de aprendizado dos otimizadores.

        Retorna:
          -  list: Lista com as taxas de aprendizado dos otimizadores.
        """
        return [DEFAULT_OPTIMIZER_GENERATOR_LEARNING,DEFAULT_OPTIMIZER_DISCRIMINATOR_LEARNING,DEFAULT_CONDITIONAL_GAN_ADAM_LEARNING_RATE,DEFAULT_CONDITIONAL_GAN_RMS_PROP_LEARNING_RATE,DEFAULT_CONDITIONAL_GAN_ADA_DELTA_LEARNING_RATE]

    def compile(self, optimizer_generator, optimizer_discriminator, loss_generator, loss_discriminator, *args, **kwargs):
        """
        Compila o modelo adversarial com os otimizadores e funções de perda fornecidos.

        Parâmetros:
           - optimizer_generator (: Otimizador para o gerador.
           - optimizer_discriminator : Otimizador para o discriminador.
           - loss_generator : Função de perda para o gerador.
           - loss_discriminator : Função de perda para o discriminador.
        """
        super().compile(*args, **kwargs)
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator
        self.loss_generator = loss_generator
        self.loss_discriminator = loss_discriminator

    @tf.function
    def train_step(self, batch):
        """
        Executa uma etapa de treino para o modelo adversarial.
        Parâmetros:
          -  batch : Um lote de dados contendo características reais e rótulos reais.

        Retorna:
          -  dict: Dicionário contendo as perdas do discriminador e do gerador.
        """
        real_feature, real_samples_label = batch
        batch_size = tf.shape(real_feature)[0]
        real_samples_label = tf.expand_dims(real_samples_label, axis=-1)
        latent_space = tf.random.normal(shape=(batch_size, self.latent_dimension))
        synthetic_feature = self.generator([latent_space, real_samples_label], training=False)

        with tf.GradientTape() as discriminator_gradient:
            label_predicted_real = self.discriminator([real_feature, real_samples_label], training=True)
            label_predicted_synthetic = self.discriminator([synthetic_feature, real_samples_label], training=True)
            label_predicted_all_samples = tf.concat([label_predicted_real, label_predicted_synthetic], axis=0)
            list_all_labels_predicted = [tf.zeros_like(label_predicted_real), tf.ones_like(label_predicted_synthetic)]
            tensor_labels_predicted = tf.concat(list_all_labels_predicted, axis=0)

            smooth_tensor_real_data = 0.15 * tf.random.uniform(tf.shape(label_predicted_real))
            smooth_tensor_synthetic_data = -0.15 * tf.random.uniform(tf.shape(label_predicted_synthetic))
            tensor_labels_predicted += tf.concat([smooth_tensor_real_data, smooth_tensor_synthetic_data], axis=0)
            loss_value = self.loss_discriminator(tensor_labels_predicted, label_predicted_all_samples)

        gradient_tape_loss = discriminator_gradient.gradient(loss_value, self.discriminator.trainable_variables)
        self.optimizer_discriminator.apply_gradients(zip(gradient_tape_loss, self.discriminator.trainable_variables))

        with tf.GradientTape() as generator_gradient:
            latent_space = tf.random.normal(shape=(batch_size, self.latent_dimension))
            synthetic_feature = self.generator([latent_space, real_samples_label], training=True)
            predicted_labels = self.discriminator([synthetic_feature, real_samples_label], training=False)
            total_loss_g = self.loss_generator(tf.zeros_like(predicted_labels), predicted_labels)

        gradient_tape_loss = generator_gradient.gradient(total_loss_g, self.generator.trainable_variables)
        self.optimizer_generator.apply_gradients(zip(gradient_tape_loss, self.generator.trainable_variables))

        return {"loss_d": loss_value, "loss_g": total_loss_g}

    def save_models(self, path_output, k_fold):
        """
        Salva os modelos do gerador e discriminador em arquivos JSON e H5.

        Parâmetros:
           - path_output : Caminho de saída para salvar os modelos.
           - k_fold : Número do fold atual.
        """
        try:
            logging.info("Saving Adversarial Model:")
            path_directory = os.path.join(path_output, self.models_saved_path)
            Path(path_directory).mkdir(parents=True, exist_ok=True)

            discriminator_file_name = self.file_name_discriminator + "_" + str(k_fold)
            generator_file_name = self.file_name_generator + "_" + str(k_fold)

            path_model = os.path.join(path_directory, "fold_" + str(k_fold + 1))
            Path(path_model).mkdir(parents=True, exist_ok=True)

            discriminator_file_name = os.path.join(path_model, discriminator_file_name)
            generator_file_name = os.path.join(path_model, generator_file_name)

            discriminator_model_json = self.discriminator.to_json()
            with open(discriminator_file_name + ".json", "w") as json_file:
                json_file.write(discriminator_model_json)
            self.discriminator.save_weights(discriminator_file_name + ".h5")

            generator_model_json = self.generator.to_json()
            with open(generator_file_name + ".json", "w") as json_file:
                json_file.write(generator_model_json)
            self.generator.save_weights(generator_file_name + ".h5")

            logging.info("  Discriminator output: {}".format(discriminator_file_name))
            logging.info("  Generator output: {}".format(generator_file_name))

        except FileExistsError:
            logging.error("File model exists")
            exit(-1)

    def load_models(self, path_output, k_fold):
        """
        Carrega os modelos gerador e discriminador de arquivos JSON e H5.

        Parâmetros:
           - path_output : Caminho de saída onde os modelos estão salvos.
           - k_fold : Número do fold atual.
        """
        try:
            logging.info("Loading Adversarial Model:")
            path_directory = os.path.join(path_output, self.models_saved_path)

            discriminator_file_name = self.file_name_discriminator + "_" + str(k_fold + 1)
            generator_file_name = self.file_name_generator + "_" + str(k_fold + 1)

            discriminator_file_name = os.path.join(path_directory, discriminator_file_name)
            generator_file_name = os.path.join(path_directory, generator_file_name)

            discriminator_model_json_pointer = open(discriminator_file_name + ".json", 'r')
            discriminator_model_json = discriminator_model_json_pointer.read()
            discriminator_model_json_pointer.close()

            self.discriminator = model_from_json(discriminator_model_json)
            self.discriminator.load_weights(discriminator_file_name + ".h5")

            generator_model_json_pointer = open(generator_file_name + ".json", 'r')
            generator_model_json = generator_model_json_pointer.read()
            generator_model_json_pointer.close()

            self.generator = model_from_json(generator_model_json)
            self.generator.load_weights(generator_file_name + ".h5")

            logging.info("Model loaded: {}".format(discriminator_file_name))
            logging.info("Model loaded: {}".format(generator_file_name))

        except FileNotFoundError:
            logging.error("Forneça um modelo existente e válido")
            exit(-1)

    def set_generator(self, generator):
        """
        Define o modelo gerador para o modelo adversarial.

        Parâmetros:
           - generator : O modelo gerador.
        """
        self.generator = generator

    def set_discriminator(self, discriminator):
        """
        Define o modelo discriminador para o modelo adversarial.

        Parâmetros:
          - discriminator : O  modelo discriminador.
        """
        self.discriminator = discriminator

    def set_latent_dimension(self, latent_dimension):
        """
        Define a dimensão do espaço latente.

        Parâmetros:
           - latent_dimension : A  dimensão do espaço latente.
        """
        self.latent_dimension = latent_dimension

    def set_optimizer_generator(self, optimizer_generator):
        """
        Define o otimizador para o gerador.

        Parâmetros:
            - optimizer_generator : O otimizador para o gerador.
        """
        self.optimizer_generator = optimizer_generator

    def set_optimizer_discriminator(self, optimizer_discriminator):
        """
        Define o otimizador para o discriminador.

        Parâmetros:
           - optimizer_discriminator : O  otimizador para o discriminador.
        """
        self.optimizer_discriminator = optimizer_discriminator

    def set_loss_generator(self, loss_generator):
        """
        Define a função de perda para o gerador.

        Parâmetros:
           - loss_generator : A função de perda para o gerador.
        """
        self.loss_generator = loss_generator

    def set_loss_discriminator(self, loss_discriminator):
        """
        Define a função de perda para o discriminador.

        Parâmetros:
           - loss_discriminator : A função de perda para o discriminador.
        """
        self.loss_discriminator = loss_discriminator

    def get_optimizer(self, training_algorithm, first_arg=None, second_arg=None):
        """
        Retorna o otimizador baseado no algoritmo de treinamento especificado.

        Parâmetros:
           - training_algorithm : O algoritmo de treinamento que deve ser utilizado ('Adam', 'RMSprop' ou 'Adadelta').
           - first_arg : Taxa de aŕendizado.
           - second_arg : Taxa de decaimento.

        Retorna:
          -  Optimizer: O otimizador especificado.

        Lança:
          -  ValueError: Se o algoritmo de treinamento fornecido não for válido.
        """
        if training_algorithm == 'Adam':
            if first_arg is None:
                first_arg = self.conditional_gan_adam_learning_rate
            if second_arg is None:
                second_arg = self.conditional_gan_adam_beta
            return Adam(first_arg, second_arg)
        elif training_algorithm == 'RMSprop':
            if first_arg is None:
                first_arg = self.conditional_gan_rms_prop_learning_rate
            if second_arg is None:
                second_arg = self.conditional_gan_rms_prop_decay_rate
            return RMSprop(first_arg, second_arg)
        elif training_algorithm == 'Adadelta':
            if first_arg is None:
                first_arg = self.conditional_gan_ada_delta_learning_rate
            if second_arg is None:
                second_arg = self.conditional_gan_ada_delta_decay_rate
            return Adadelta(first_arg, second_arg)
        else:
            raise ValueError("Algoritmo de treinamento inválido. Use 'Adam', 'RMSprop' ou 'Adadelta'.")
