
"""
Este módulo  contém  as definições das classes das ferramentas de plot utilizadas para geração dos gráficos. 
Clases:
- PlotConfusionMatrix (Classe responsável pelo plot das matrizes de confusão).
- PlotCurveLoss (Classe responsável pelos plots das curvas de perda durante o treinamento de modelos generativos adversariais).
- PlotClassificationMetrics (Classe responsável pelo plot das métricas de utilidade dos classificadores).
- PlotFidelityeMetrics (Classe responsável pelo plot das métricas de fidelidade).
- ProbabilisticMetrics (Classe responsável pelo cálculo de métricas de probabilidade entre rótulos reais e os valores preditos).
"""
# Importação de bibliotecas necessárias
import os
import numpy as np
import itertools
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.special import rel_entr
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import pairwise

# Definição de constantes padrão para a matriz de confusão e curvas de perda
DEFAULT_MATRIX_CONFUSION_CLASS_LABELS = ["Maligno", "Benigno"]
DEFAULT_MATRIX_CONFUSION_PREDICT_LABELS = ["Rótulo Verdadeiro", "Rótulo Predito"]
#Título padrão para as matrizes de confusão
DEFAULT_MATRIX_CONFUSION_TITLE = "Matriz de Confusão"
# Valor padrão para a largura das barras
DEFAULT_WIDTH_BAR = 0.2
# Valor padrão para a fonte dos plots
DEFAULT_FONT_SIZE = 12
# Valor padrão para o declínio das legendas do plot
DEFAULT_MATRIX_CONFUSION_ROTATION_LEGENDS = 45
#Valores padrões ara as legendas dos plots
DEFAULT_LOSS_CURVE_LEGEND_GENERATOR = "Gerador"
DEFAULT_LOSS_CURVE_LEGEND_DISCRIMINATOR = "Discriminador"
DEFAULT_LOSS_CURVE_LEGEND_ITERATIONS = "Interações (Épocas)"
DEFAULT_LOSS_CURVE_TITLE_PLOT = "Perda do Gerador e Discriminador"
DEFAULT_LOSS_CURVE_LEGEND_LOSS = "Perda"
DEFAULT_LOSS_CURVE_LEGEND_NAME = "Legenda"
DEFAULT_LOSS_CURVE_PREFIX_FILE = "curve_training_error"
DEFAULT_TITLE_COMPARATIVE_PLOTS = "Comparativo entre dados sintéticos e reais (Média)"
DEFAULT_PLOT_CLASSIFIER_METRICS_LABELS = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
DEFAULT_PLOT_FIDELITY_METRICS_LABELS = ['Similaridade de Cossenos', 'Erro Médio Quadrático', 'Máxima Discrepância Média']

# Cores padrão para gráficos
DEFAULT_COLOR_MAP = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22','#17becf']
DEFAULT_COLOR_MAP_REGRESSIVE = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f' ]
DEFAULT_COLOR_NAME = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd','PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

class PlotConfusionMatrix:
    """
   Classe responsável pelo plot das matrizes de confusão.
    
    Funções:
        - __init__: Inicializa a classe com valores padrão ou fornecidos.
        - plot_confusion_matrix: Plota a matriz de confusão.
        - set_class_labels: Define os rótulos das classes.
        - set_titles_confusion_matrix: Define os títulos dos eixos da matriz de confusão.
        - set_title_confusion_matrix: Define o título da matriz de confusão.
        - set_legend_rotation: Define a rotação da legenda da matriz de confusão.
    """
    def __init__(self, class_labels=None, titles_confusion_matrix_labels=None,
                 title_confusion_matrix=DEFAULT_MATRIX_CONFUSION_TITLE,
                 legend_rotation=DEFAULT_MATRIX_CONFUSION_ROTATION_LEGENDS):
        """
        Inicializa a classe PlotConfusionMatrix com parâmetros padrão ou fornecidos.

        Parâmetros:
            - class_labels: Lista de rótulos de classe.
            - titles_confusion_matrix_labels: Títulos dos eixos da matriz de confusão.
            - title_confusion_matrix: Título da matriz de confusão.
            - legend_rotation: Rotação da legenda da matriz de confusão.
        """
        if titles_confusion_matrix_labels is None:
            titles_confusion_matrix_labels = DEFAULT_MATRIX_CONFUSION_PREDICT_LABELS

        if class_labels is None:
            class_labels = DEFAULT_MATRIX_CONFUSION_CLASS_LABELS

        self.class_labels = class_labels
        self.titles_confusion_matrix = titles_confusion_matrix_labels
        self.title_confusion_matrix = title_confusion_matrix
        self.legend_rotation = legend_rotation

    def plot_confusion_matrix(self, confusion_matrix, confusion_matrix_title=None, cmap=None):
        """
        Plota a matriz de confusão.

        Parâmetros:
            - confusion_matrix: A matriz de confusão a ser plotada.
            - confusion_matrix_title: Título da matriz de confusão.
            - cmap: Mapa de cores a ser usado no plot.
        """
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)

        if confusion_matrix_title is None:
            confusion_matrix_title = self.title_confusion_matrix

        plt.title(confusion_matrix_title)
        plt.colorbar()
        tick_marks = np.arange(len(self.class_labels))
        plt.xticks(tick_marks, self.class_labels, rotation=self.legend_rotation)
        plt.yticks(tick_marks, self.class_labels)
        thresh = confusion_matrix.max() / 2.

        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            plt.text(j, i, confusion_matrix[i, j], horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel(self.titles_confusion_matrix[0], fontsize=12)
        plt.xlabel(self.titles_confusion_matrix[1], fontsize=12)

    def set_class_labels(self, class_labels):
        """
        Define os rótulos das classes da matriz de confusão.

        Parâmetros:
            - class_labels: Lista de novos rótulos de classe.
        """
        self.class_labels = class_labels

    def set_titles_confusion_matrix(self, titles_confusion_matrix):
        """
        Define os títulos dos eixos da matriz de confusão.

        Parâmetros:
            - titles_confusion_matrix: Lista com os novos títulos dos eixos.
        """
        self.titles_confusion_matrix = titles_confusion_matrix

    def set_title_confusion_matrix(self, title_confusion_matrix):
        """
        Define o título da matriz de confusão.

        Parâmetros:
            - title_confusion_matrix: Novo título para a matriz de confusão.
        """
        self.title_confusion_matrix = title_confusion_matrix

    def set_legend_rotation(self, legend_rotation):
        """
        Define a rotação da legenda da matriz de confusão.

        Parâmetros:
            - legend_rotation: Novo valor para a rotação da legenda.
        """
        self.legend_rotation = legend_rotation

class PlotCurveLoss:
    """
    Classe responsável pelos plots das curvas de perda durante o treinamento de modelos generativos adversariais.
        Funções:
            - __init__: Inicializa a classe com valores padrão ou fornecidos.
            - plot_training_loss_curve: Plota a curva de perda de treinamento.
            - set_loss_curve_legend_generator: Define a legenda para a curva de perda do gerador.
            - set_loss_curve_legend_discriminator: Define a legenda para a curva de perda do discriminador.
            - set_loss_curver_title_plot: Define o título do plot da curva de perda.
            - set_loss_curve_legend_iterations: Define a legenda para as iterações (épocas) no plot.
            - set_loss_curve_legend_loss: Define a legenda para a perda no plot.
            - set_loss_curve_legend_name: Define o nome da legenda no plot.
            - set_loss_curve_prefix_file: Define o prefixo do nome do arquivo para salvar o plot.
  
    """

    def __init__(self, loss_curve_legend_generator=DEFAULT_LOSS_CURVE_LEGEND_GENERATOR,
                 loss_curve_legend_discriminator=DEFAULT_LOSS_CURVE_LEGEND_DISCRIMINATOR,
                 loss_curver_title_plot=DEFAULT_LOSS_CURVE_TITLE_PLOT,
                 loss_curve_legend_iterations=DEFAULT_LOSS_CURVE_LEGEND_ITERATIONS,
                 loss_curve_legend_loss=DEFAULT_LOSS_CURVE_LEGEND_LOSS,
                 loss_curve_legend_name=DEFAULT_LOSS_CURVE_LEGEND_NAME,
                 loss_curve_prefix_file=DEFAULT_LOSS_CURVE_PREFIX_FILE):
        """
        Inicializa a classe PlotCurveLoss com parâmetros padrão ou fornecidos.

        Parâmetros:
            - loss_curve_legend_generator: Legenda para a curva de perda do gerador.
            - loss_curve_legend_discriminator: Legenda para a curva de perda do discriminador.
            - loss_curver_title_plot: Título do plot da curva de perda.
            - loss_curve_legend_iterations: Legenda para as iterações (épocas).
            - loss_curve_legend_loss: Legenda para a perda.
            - loss_curve_legend_name: Nome da legenda.
            - loss_curve_prefix_file: Prefixo do nome do arquivo para salvar o plot.
        """
        self.loss_curve_legend_generator = loss_curve_legend_generator
        self.loss_curve_legend_discriminator = loss_curve_legend_discriminator
        self.loss_curver_title_plot = loss_curver_title_plot
        self.loss_curve_legend_iterations = loss_curve_legend_iterations
        self.loss_curve_legend_loss = loss_curve_legend_loss
        self.loss_curve_legend_name = loss_curve_legend_name
        self.loss_curve_prefix_file = loss_curve_prefix_file

    def plot_training_loss_curve(self, generator_loss, discriminator_loss, output_dir, k_fold, path_curve_loss):
        """
        Plota a curva de perda de treinamento para o gerador e discriminador e a salva como um arquivo PDF.

        Parâmetros:
            - generator_loss: Lista contendo as perdas do gerador ao longo das épocas.
            - discriminator_loss: Lista contendo as perdas do discriminador ao longo das épocas.
            - output_dir: Diretório onde o arquivo de saída será salvo.
            - k_fold: Índice do fold atual (para k-fold cross-validation).
            - path_curve_loss: Caminho adicional dentro do diretório de saída para salvar o arquivo.
        """
        if output_dir is not None:
            new_loss_curve_plot = go.Figure()
            new_loss_curve_plot.add_trace(go.Scatter(x=list(range(len(generator_loss))), y=generator_loss,
                                                     name=self.loss_curve_legend_generator))
            new_loss_curve_plot.add_trace(go.Scatter(x=list(range(len(discriminator_loss))), y=discriminator_loss,
                                                     name=self.loss_curve_legend_discriminator))

            new_loss_curve_plot.update_layout(title=self.loss_curver_title_plot,
                                              xaxis_title=self.loss_curve_legend_iterations,
                                              yaxis_title=self.loss_curve_legend_loss,
                                              legend_title=self.loss_curve_legend_name)

            Path(os.path.join(output_dir, path_curve_loss)).mkdir(parents=True, exist_ok=True)
            file_name_output = self.loss_curve_prefix_file + "_k_{}.pdf".format(str(k_fold + 1))
            pio.write_image(new_loss_curve_plot, os.path.join(output_dir, path_curve_loss, file_name_output))


    def set_loss_curve_legend_generator(self, loss_curve_legend_generator):
        """
        Define a legenda para a curva de perda do gerador.

        Parâmetros:
            - loss_curve_legend_generator: Valor para a legenda do gerador.
        """
        self.loss_curve_legend_generator = loss_curve_legend_generator

    def set_loss_curve_legend_discriminator(self, loss_curve_legend_discriminator):
        """
        Define a legenda para a curva de perda do discriminador.

        Parâmetros:
            - loss_curve_legend_discriminator: Valor para a legenda do discriminador.
        """
        self.loss_curve_legend_discriminator = loss_curve_legend_discriminator

    def set_loss_curver_title_plot(self, loss_curver_title_plot):
        """
        Define o título do plot da curva de perda.

        Parâmetros:
            - loss_curver_title_plot: Valor para o título do plot.
        """
        self.loss_curver_title_plot = loss_curver_title_plot

    def set_loss_curve_legend_iterations(self, loss_curve_legend_iterations):
        """
        Define a legenda para as iterações (épocas) no plot da curva de perda.

        Parâmetros:
            - loss_curve_legend_iterations: Valor para a legenda das iterações.
        """
        self.loss_curve_legend_iterations = loss_curve_legend_iterations

    def set_loss_curve_legend_loss(self, loss_curve_legend_loss):
        """
        Define a legenda para a perda no plot da curva de perda.

        Parâmetros:
            - loss_curve_legend_loss: Valor para a legenda da perda.
        """
        self.loss_curve_legend_loss = loss_curve_legend_loss

    def set_loss_curve_legend_name(self, loss_curve_legend_name):
        """
        Define o nome da legenda no plot da curva de perda.

        Parâmetros:
            - loss_curve_legend_name: valor para o nome da legenda.
        """
        self.loss_curve_legend_name = loss_curve_legend_name

    def set_loss_curve_prefix_file(self, loss_curve_prefix_file):
        """
        Define o prefixo do nome do arquivo para salvar o plot da curva de perda.

        Parâmetros:
            - loss_curve_prefix_file: Valor para o prefixo do nome do arquivo.
        """
        self.loss_curve_prefix_file = loss_curve_prefix_file

class PlotClassificationMetrics:
    """
    Classe responsável pelo plot das métricas de utilidade dos classificadores
    
    Funções:
        - __init__: Inicializa a classe com valores padrão ou fornecidos.
        - plot_classifier_metrics: Plota as métricas de classificação.
        - set_labels_bar_metrics: Define os rótulos das barras de métricas.
        - set_color_map_bar: Define o mapa de cores para as barras.
        - set_width_bar: Define a largura das barras.
        - set_font_size: Define o tamanho da fonte das anotações.
    """


    def __init__(self, labels_bar_metrics=None, color_map_bar=None, width_bar=DEFAULT_WIDTH_BAR, font_size=DEFAULT_FONT_SIZE):
        """
        Inicializa a classe PlotClassificationMetrics com parâmetros padrão ou fornecidos.

        Parâmetros:
            - labels_bar_metrics: Lista de rótulos para as barras de métricas.
            - color_map_bar: Mapa de cores para as barras.
            - width_bar: Largura das barras.
            - font_size: Tamanho da fonte das anotações.
        """
        if color_map_bar is None:
            color_map_bar = DEFAULT_COLOR_MAP

        if labels_bar_metrics is None:
            labels_bar_metrics = DEFAULT_PLOT_CLASSIFIER_METRICS_LABELS

        self.labels_bar_metrics = labels_bar_metrics
        self.color_map_bar = color_map_bar
        self.width_bar = width_bar
        self.font_size = font_size

    def plot_classifier_metrics(self, classifier_type, accuracy_list, precision_list, recall_list, f1_score_list, plot_filename, plot_title, type_of_classifier):
        """
        Plota as métricas de classificação.

        Parâmetros:
            - classifier_type: Tipo de classificador.
            - accuracy_list: Lista de valores de acurácia.
            - precision_list: Lista de valores de precisão.
            - recall_list: Lista de valores de recall.
            - f1_score_list: Lista de valores de F1-Score.
            - plot_filename: Nome do arquivo para salvar o plot.
            - plot_title: Título do plot.
            - type_of_classifier: Tipo de classificador sendo plotado (TR_As,TR_As) .
        """
        list_all_metrics = [accuracy_list, precision_list, recall_list, f1_score_list]
        new_plot_bars = go.Figure()
        color_map = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        if type_of_classifier == "TR_As":
            metrics_name = ['Acurácia-TR_As', 'Precisão-TR_As', 'Recall-TR_As', 'F1-Score-TR_As']
        else:
            metrics_name = ['Acurácia-TS_Ar', 'Precisão-TS_Ar', 'Recall-TS_Ar', 'F1-Score-TS_Ar']
        # Itera sobre as métricas,cores e rótulos 
        for metric, metric_values, color in zip(metrics_name, list_all_metrics, color_map):
            try:
               #Cálculo das média e desvio padrão das métricas
                metric_mean = np.mean(metric_values)
                metric_std = np.std(metric_values)
                # Adiciona as barras com as métricas
                new_plot_bars.add_trace(go.Bar(
                    x=[metric], y=[metric_mean], name=metric, marker=dict(color=color),
                    error_y=dict(type='constant', value=metric_std, visible=True),
                    width=self.width_bar
                ))

                new_plot_bars.add_annotation(
                    x=metric, y=metric_mean + metric_std, xref="x", yref="y",
                    text=f' {metric_std:.4f}', showarrow=False,
                    font=dict(color='black', size=self.font_size),
                    xanchor='center', yanchor='bottom'
                )
            except Exception as e:
                print(f"Metric {metric} error: {e}")

        y_label_dictionary = dict(
            title=f'Média {len(accuracy_list)} dobras', tickmode='linear', tick0=0.0, dtick=0.1,
            gridcolor='black', gridwidth=.05
        )

        new_plot_bars.update_layout(
            barmode='group', title=plot_title, yaxis=y_label_dictionary,
            xaxis=dict(title=f'Desempenho com {classifier_type}'), showlegend=False,
            plot_bgcolor='white'
        )

        pio.write_image(new_plot_bars, plot_filename)

    def set_labels_bar_metrics(self, labels_bar_metrics):
        """
        Define os rótulos das barras de métricas.

        Parâmetros:
            - labels_bar_metrics: Lista de rótulos para as barras.
        """
        self.labels_bar_metrics = labels_bar_metrics

    def set_color_map_bar(self, color_map_bar):
        """
        Define o mapa de cores para as barras.

        Parâmetros:
            - color_map_bar: Mapa de cores para as barras.
        """
        self.color_map_bar = color_map_bar

    def set_width_bar(self, width_bar):
        """
        Define a largura das barras.

        Parâmetros:
            - width_bar: Largura das barras.
        """
        self.width_bar = width_bar

    def set_font_size(self, font_size):
        """
        Define o tamanho da fonte das anotações.

        Parâmetros:
            - font_size: Tamanho da fonte das anotações.
        """
        self.font_size = font_size

class PlotFidelityeMetrics:
    """
    Classe para gerar e plotar métricas de fidelidade.
    
    Funções:
        - __init__: Inicializa a classe com valores padrão ou fornecidos.
        - plot_fidelity_metrics: Plota as métricas de fidelidade.
        - set_labels_bar_metrics: Define os rótulos das barras de métricas de fidelidade.
        - set_color_map_bar: Define o mapa de cores para as barras.
        - set_width_bar: Define a largura das barras.
        - set_font_size: Define o tamanho da fonte das anotações.
    """

    def __init__(self, labels_plot_fidelity_metrics=None, color_map_bar=None, width_bar=DEFAULT_WIDTH_BAR,
                 font_size=DEFAULT_FONT_SIZE, plot_title=DEFAULT_TITLE_COMPARATIVE_PLOTS):
        """
        Inicializa a classe PlotFidelityeMetrics com parâmetros padrão ou fornecidos.

        Parâmetros:
            - labels_plot_fidelity_metrics: Lista de rótulos para as barras de métricas de fidelidade.
            - color_map_bar: Mapa de cores para as barras.
            - width_bar: Largura das barras.
            - font_size: Tamanho da fonte das anotações.
            - plot_title: Título do plot.
        """
        if color_map_bar is None:
            color_map_bar = DEFAULT_COLOR_MAP_REGRESSIVE

        if labels_plot_fidelity_metrics is None:
            labels_plot_fidelity_metrics = DEFAULT_PLOT_FIDELITY_METRICS_LABELS

        self.labels_plot_fidelity_metrics = labels_plot_fidelity_metrics
        self.color_map_bar = color_map_bar
        self.width_bar = width_bar
        self.plot_title_axis_x = plot_title
        self.font_size = font_size

    def plot_fidelity_metrics(self, mean_squared_error_list, list_cosine_similarity, list_max_mean_discrepancy, plot_filename, plot_title):
        """
        Realiza o plot das métricas de fidelidade.

        Parâmetros:
            - mean_squared_error_list: Lista de valores de erro médio quadrático.
            - list_cosine_similarity: Lista de valores de similaridade de cossenos.
            - list_max_mean_discrepancy: Lista de valores de máxima discrepância média.
            - plot_filename: Nome do arquivo para salvar o plot.
            - plot_title: Título do plot.
        """
        list_metrics = [list_cosine_similarity, mean_squared_error_list, list_max_mean_discrepancy]
        new_plot_bars = go.Figure()
        # Itera sobre as métricas,cores e rótulos 
        for metric, metric_values, color in zip(self.labels_plot_fidelity_metrics, list_metrics, self.color_map_bar):
            try:
                #Cálculo das média e desvio padrão das métricas
                metric_mean = np.mean(metric_values)
                metric_std = np.std(metric_values)
                # Adiciona as barras com as métricas
                new_plot_bars.add_trace(go.Bar(
                    x=[metric], y=[metric_mean], name=metric, marker=dict(color=color),
                    error_y=dict(type='constant', value=metric_std, visible=True),
                    width=self.width_bar
                ))

                new_plot_bars.add_annotation(
                    x=metric, y=metric_mean + metric_std, xref="x", yref="y",
                    text=f' {metric_std:.4f}', showarrow=False,
                    font=dict(color='black', size=self.font_size),
                    xanchor='center', yanchor='bottom'
                )
            except Exception as e:
                print(f"Metric: {metric} Exception: {e}")

        y_label_dictionary = dict(
            title=f'Média {len(mean_squared_error_list)} dobras', tickmode='linear', tick0=0.0, dtick=0.1,
            gridcolor='black', gridwidth=.05
        )

        new_plot_bars.update_layout(
            barmode='group', title=plot_title, yaxis=y_label_dictionary,
            xaxis=dict(title=self.plot_title_axis_x), showlegend=False,
            plot_bgcolor='white'
        )

        pio.write_image(new_plot_bars, plot_filename)

    def set_labels_bar_metrics(self, labels_bar_metrics):
        """
        Define os rótulos das barras de métricas de fidelidade.

        Parâmetros:
            - labels_bar_metrics: Lista de rótulos para as barras.
        """
        self.labels_plot_fidelity_metrics = labels_bar_metrics

    def set_color_map_bar(self, color_map_bar):
        """
        Define o mapa de cores para as barras.

        Parâmetros:
            - color_map_bar: Mapa de cores para as barras.
        """
        self.color_map_bar = color_map_bar

    def set_width_bar(self, width_bar):
        """
        Define a largura das barras.

        Parâmetros:
            - width_bar: Largura das barras.
        """
        self.width_bar = width_bar

    def set_font_size(self, font_size):
        """
        Define o tamanho da fonte das anotações.

        Parâmetros:
            - font_size: Tamanho da fonte das anotações.
        """
        self.font_size = font_size

class ProbabilisticMetrics:
    """
    Classe responsável pelo cálculo de métricas de probabilidade entre rótulos reais e os valores preditos.

    Funções:
        - __init__: Inicializa a classe.
        - get_mean_squared_error: Calcula o erro quadrático médio.
        - get_cosine_similarity: Calcula a similaridade de cossenos.
        - get_kl_divergence: Calcula a divergência de Kullback-Leibler.
        - get_maximum_mean_discrepancy: Calcula a discrepância média máxima.
        - get_accuracy: Calcula a acurácia.
        - get_precision: Calcula a precisão.
        - get_recall: Calcula o recall.
        - get_f1_score: Calcula o F1-Score.
    """

    def __init__(self):
        """
        Inicializa a classe ProbabilisticMetrics.
        """
        pass

    @staticmethod
    def get_mean_squared_error(real_label, predicted_label):
        """
        Calcula o erro quadrático médio entre os rótulos reais e previstos.

        Parâmetros:
            - real_label: Lista ou array de rótulos reais.
            - predicted_label: Lista ou array de rótulos previstos.

        Retorna:
            - Erro quadrático médio.
        """
        return mean_squared_error(real_label, predicted_label)

    @staticmethod
    def get_cosine_similarity(real_label, predicted_label):
        """
        Calcula a similaridade de cossenos média entre os rótulos reais e previstos.

        Parâmetros:
            - real_label: Lista ou array de rótulos reais.
            - predicted_label: Lista ou array de rótulos previstos.

        Retorna:
            - Similaridade de cossenos média.
        """
        return sum(pairwise.cosine_similarity(real_label, predicted_label)[0]) / len(real_label)

    @staticmethod
    def get_kl_divergence(real_label, predicted_label):
        """
        Calcula a divergência de Kullback-Leibler entre os rótulos reais e previstos.

        Parâmetros:
            - real_label: Lista ou array de rótulos reais.
            - predicted_label: Lista ou array de rótulos previstos.

        Retorna:
            - Divergência de Kullback-Leibler.

        Observação:
            - Pequeno valor (1e-10) é adicionado para evitar problemas de log(0).
        """
        real_label = np.asarray(real_label, dtype=np.float32) + 1e-10
        predicted_label = np.asarray(predicted_label, dtype=np.float32) + 1e-10
        return sum(rel_entr(real_label, predicted_label))

    @staticmethod
    def get_maximum_mean_discrepancy(real_label, predicted_label):
        """
        Calcula a discrepância média máxima entre os rótulos reais e previstos.

        Parâmetros:
            - real_label: Lista ou array de rótulos reais.
            - predicted_label: Lista ou array de rótulos previstos.

        Retorna:
            - Discrepância média máxima.
        """
        delta = real_label.mean(0) - predicted_label.mean(0)
        return delta.dot(delta.T)

    @staticmethod
    def get_accuracy(real_label, predicted_label):
        """
        Calcula a acurácia entre os rótulos reais e previstos.

        Parâmetros:
            - real_label: Lista ou array de rótulos reais.
            - predicted_label: Lista ou array de rótulos previstos.

        Retorna:
            - Acurácia.
        """
        return accuracy_score(real_label, predicted_label)

    @staticmethod
    def get_precision(real_label, predicted_label):
        """
        Calcula a precisão entre os rótulos reais e previstos.

        Parâmetros:
            - real_label: Lista ou array de rótulos reais.
            - predicted_label: Lista ou array de rótulos previstos.

        Retorna:
            - Precisão.
        """
        return precision_score(real_label, predicted_label)

    @staticmethod
    def get_recall(real_label, predicted_label):
        """
        Calcula o recall entre os rótulos reais e previstos.

        Parâmetros:
            - real_label: Lista ou array de rótulos reais.
            - predicted_label: Lista ou array de rótulos previstos.

        Retorna:
            - Recall.
        """
        return recall_score(real_label, predicted_label)

    @staticmethod
    def get_f1_score(real_label, predicted_label):
        """
        Calcula o F1-Score entre os rótulos reais e previstos.

        Parâmetros:
            - real_label: Lista ou array de rótulos reais.
            - predicted_label: Lista ou array de rótulos previstos.

        Retorna:
            - F1-Score.
        """
        return f1_score(real_label, predicted_label)
