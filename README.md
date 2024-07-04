## Preparação e Execução

1. Clonar o repositório 
   ```bash
    git clone https://github.com/MalwareDataLab/DroidAugmentor.git
    cd DroidAugmentor
   ```

   
    **Clonar o repositorio com os datasets**
      ```bash
       git clone https://github.com/MalwareDataLab/Datasets.git
      ```

2. Executar a demonstração de funcionamento da ferramenta: 

   **Opção 1**: instalar as dependências e executar a aplicação em um ambiente Linux.
   ```bash
   ./run_demo_app.sh
   ```

   **Opção 2**: baixar a imagem do [hub.docker.com](hub.docker.com) e instanciar um container Docker.
   ```bash
   ./run_demo_docker.sh
   ```
     
   **Opção 3**: construir uma imagem Docker localmente a partir do Dockerfile e instanciar um container.
   
   ```bash
   ./scripts/docker_build.sh
   ./scripts/docker_run_solo.sh
   ```

    
4. Executar os mesmos experimentos (campanhas) do paper

   ```bash
    ./run_sf23_experiments.sh
    ```

## Fluxo de execução 
![fluxograma_novo(2) CGAN data augmentor SF SBSEG24(1)](https://github.com/MalwareDataLab/DroidAugmentor/assets/72932783/0ca6203a-356d-4b7f-a79c-e16de60ff5b6)
O fluxo de execução da ferramenta consiste de três etapas:

   **Seleção de dataset**: Esta etapa iniciada através da execução script de balancemento das entradas: 
   ```bash
   python3 rebalancer.py -i arquvo_de_entrada.csv -o arquivo_de_saida.csv
   ```

   Após o balancemento o dataset será dividido em K dobras com base nos hiperparâmetros de entrada.

  **Treinamento**: Nesta etapa ocorre o treinamento da CGAN e geração dos dados sintenticos. Após a geração dos dados são treinados dois classificadores TS-AA(Treinado com dataset S, avaliado com A) e TR-As(Treinado com R, avaliado com s).

   **Avaliação**: Consiste da coleta de métricas de fidelidade e utilidade dos classificadores e dados sintéticos. E subsequente aplicação do teste de wilcoxon sobre as métricas na dobra final



## Executando os datasets balanceados
O script em bash execution.sh é reponsavel pela execução dos datasets balanceados criados apartir dos 10 datasets originais.

Executar o script: 


   ```bash
   bash execution.sh
   ```


### Executando outros experimentos

A ferramenta conta com o **run_campaign.py** para automatizar o treinamento e a avaliação da cGAN. O **run_campaign.py** permite executar várias campanhas de avaliação com diferentes parâmetros, registrando os resultados em arquivos de saída para análise posterior. O usuário poderá visualmente realizar uma análise comparativa das diferentes configurações em relação aos conjuntos de dados utilizados.

### Configurar o pipenv

```
pip install pipenv
```
```
pipenv install -r requirements.txt
```


Execução básica:
```
pipenv python3 run_campaign.py
```

Exemplo de execução de uma campanha pré-configurada:

```
pipenv run python3 run_campaign.py -c sf23_1l_256

```

Mesma campanha (sf23_1l_256) sendo executada diretamente na aplicação (**main.py**):
```
pipenv run python main.py --verbosity 20 --output_dir outputs/out_2023-08-05_12-04-18/sf23_1l_256/combination_2 --input_dataset datasets/drebin215_original_5560Malwares_6566Benign.csv --dense_layer_sizes_g 256 --dense_layer_sizes_d 256 --number_epochs 1000 --training_algorithm Adam
```

###  Parâmetros dos testes automatizados:

      --------------------------------------------------------------

    --campaign ou -c:    Especifica a campanha de avaliação que você deseja executar. 
                         Você pode fornecer o nome de uma campanha específica ou uma  
                         lista de campanhas separadas por vírgula. 
                         Por exemplo: --campaign sf23_1l_64 ou --campaign 
                         sf23_1l_64,sf23_1l_128.

    --demo ou -d:
                         Ativa o modo demo. Quando presente, o script será executado 
                         no modo demo, o que pode ter comportamento reduzido 
                         ou exibir informações de teste.
                         --verbosity ou -v: Especifica o nível de verbosidade do log.
                         Pode ser INFO (1) ou DEBUG (2). 
                         Por padrão, o nível de verbosidade é definido como INFO.


     Outros parâmetros de entrada são definidos dentro das campanhas de avaliação em 
     campaigns_available. Cada campanha tem suas próprias configurações específicas, 
     como input_dataset, number_epochs, training_algorithm, dense_layer_sizes_g, 
     dense_layer_sizes_d, classifier, activation_function, dropout_decay_rate_g, 
     dropout_decay_rate_d, e data_type. As configurações podem variar dependendo do 
     objetivo e das configurações específicas de cada campanha.  


     Em campaigns_available o script irá iterar sobre as combinações de configurações 
     especificadas e executar os experimentos correspondentes.

    --------------------------------------------------------------


## Executando a ferramenta no Google Colab

```
from google.colab import drive
drive.mount('/content/drive')
```

```
!pip install -r requirements.txt
```
```
input_file_path = "/content/dataset.csv"
```

```
!python main.py -i "$input_file_path" 
```

Obs.: Lembre-se de ter Models, Tools e a main devidamente importada no seu drive.
 <td><img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/layout/arquivos.JPG" style="max-width:100%;"></td>


## Parâmetros da ferramenta:
    --------------------------------------------------------------
   
          (main.py):

           -i ,  --input_dataset        Caminho para o arquivo do dataset real de entrada         
           -o ,  --output_dir           Diretório para gravação dos arquivos de saída.
           --data_type                  Tipo de dado para representar as características das amostras.
           --num_samples_class_malware  Número de amostras da Classe 1 (maligno).
           --num_samples_class_benign   Número de amostras da Classe 0 (benigno).
           --number_epochs              Número de épocas (iterações de treinamento) da cGAN.
           --k_fold                     Número de subdivisões da validação cruzada 
           --initializer_mean           Valor central da distribuição gaussiana do inicializador.
           --initializer_deviation      Desvio padrão da distribuição gaussiana do inicializador.
           --latent_dimension           Dimensão do espaço latente para treinamento cGAN.
           --training_algorithm         Algoritmo de treinamento para cGAN. Opções: 'Adam', 'RMSprop', 'Adadelta'.
           --activation_function        Função de ativação da cGAN. Opções: 'LeakyReLU', 'ReLU', 'PReLU'.
           --dropout_decay_rate_g       Taxa de decaimento do dropout do gerador da cGAN.
           --dropout_decay_rate_d       Taxa de decaimento do dropout do discriminador da cGAN.
           --dense_layer_sizes_g        Valores das camadas densas do gerador.
           --dense_layer_sizes_d        Valores das camadas densas do discriminador.
           --batch_size                 Tamanho do lote da cGAN.
           --verbosity                  Nível de verbosidade.
           --save_models                Opção para salvar modelos treinados.
           --path_confusion_matrix      Diretório de saída das matrizes de confusão.
           --path_curve_loss            Diretório de saída dos gráficos de curva de treinamento.
           -a,  --use_aim               Opção para utilizar a ferramenta de rastreamento Aimstack
           -ml, --use_mlflow            Opção para utilizar a ferramenta de rastreamento mlflow
           -rid, --run_id              Opção ligado ao mlflow, utilizada para resumir uma execução não terminada 
           -np', --use_neptune          Opção para utilizar a ferramenta de rastreamento Neptune 
           -tb, --use_tensorboard       Opção para utilizar a ferramenta de rastreamento Tensorboard

        --------------------------------------------------------------
        

## Ambientes de teste

A ferramenta foi executada e testada na prática nos seguintes ambientes:

1. Windows 10<br/>
   Kernel Version = 10.0.19043<br/>
   Python = 3.8.10 <br/>
   Módulos Python conforme [requirements](requirements.txt).
   
2. Linux Ubuntu 22.04.2 LTS<br/>
   Kernel Version = 5.15.109+<br/>
   Python = 3.8.10 <br/>
   Módulos Python conforme [requirements](requirements.txt).

3. Linux Ubuntu 22.04.2 LTS<br/>
   Kernel Version =  5.19.0-46-generic<br/>
   Python = 3.10.6<br/>
   Módulos Python conforme [requirements](requirements.txt).

4. Linux Ubuntu 20.04.6 LTS<br/>
   Kernel Version =  5.19.0-46-generic<br/>
   Python = 3.8.10<br/>
   Módulos Python conforme [requirements](requirements.txt).

## Link para o repositorio com os datasets
[datasets](https://github.com/MalwareDataLab/Datasets)
