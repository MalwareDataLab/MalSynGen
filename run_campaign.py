"""
Módulo responsavel pela execução de campanhas da ferramenta, incluindo as configurações de experimentos do paper.

Classes:
    IntRange :  Tipo personalizado de argparse que representa um inteiro delimitado por um intervalo.
Funções:
    - print_config : Imprime a configuração dos argumentos para fins de logging.
    - convert_flot_to_int : Converte um valor float para int multiplicando por 100.
    - run_cmd : A função executa um comando de shell especificado e registra a saída.
    - check_files : Verifica se os arquivos especificados existem.
    - main: Função principal que configura e executa as campanhas.
    
"""
# Importação de bibliotecas necessárias
try:
    import sys
    import os
    import argparse
    import logging
    import subprocess
    import shlex
    import datetime
    from logging.handlers import RotatingFileHandler
    from pathlib import Path
    import itertools
    import mlflow

#Tratamento de erro de import
except ImportError as error:
    print(error)
    print()
    print(" ")
    print()
    sys.exit(-1)

# Definindo constantes padrão

DEFAULT_VERBOSITY_LEVEL = logging.INFO

NUM_EPOCHS = 1000
TIME_FORMAT = '%Y-%m-%d_%H:%M:%S'
# Estabelece a campanha padrão como a demo
DEFAULT_CAMPAIGN = "demo"
# Caminho para os arquivos de log
PATH_LOG = 'logs'
# Caminho para os dataset
PATH_DATASETS = 'datasets'
PATHS = [PATH_LOG]
Parametros = None
#Valores para os comandos de entrada o COMMAND não possuei a opção de rastreameto do mflow, enquanto que COMMAND2 possui
COMMAND = "pipenv run python main.py   "
COMMAND2 = "pipenv run python main.py -ml  "

#Dataset utiliados
datasets = ['datasets/kronodroid_emulador-balanced.csv', 'datasets/kronodroid_real_device-balanced.csv']

def list_of_ints(arg):
    return list(map(int, arg.split(',')))
def list_of_floats(arg):
    return list(map(float, arg.split(',')))
# Define a custom argument type for a list of integers
def list_of_strs(arg):
    return list(map(str, arg.split(',')))

# Definindo campanhas disponíveis
"""
  Campanhas:
   - Demo: execução do demo proposto no arquivo run_demo_venv.sh
   - Demo2: execução de um demo alternativo que engloba ambos datasets.
   - Kronodroid_r: Mesma configuração do paper para o dataset Kronodroid_r.
   - Kronodroid_E: Mesma configuração do paper para o dataset Kronodroid_E.
   - SF24_4096_2048_10: Mesma configuração dos experimentos dos papers

"""
campaigns_available = {
    'demo': {
        'input_dataset': ['datasets/kronodroid_emulador-balanced.csv'],
        "num_samples_class_benign": ['10000'],
        "num_samples_class_malware": ['10000'],
        'number_epochs': ['100'],
        'k_fold': ['2'],
        'training_algorithm': ['Adam'],
    },
    'Kronodroid_r': {
        'input_dataset': ['datasets/kronodroid_emulador-balanced.csv'],
        "dense_layer_sizes_g": ['4096'],
        "dense_layer_sizes_d": ['2048'],
        "num_samples_class_benign": ['10000'],
        "num_samples_class_malware": ['10000'],
        'number_epochs': ['500'],
        'k_fold': ['10'],
        'training_algorithm': ['Adam'],
    },
    'Kronodroid_e': {
        'input_dataset': ['datasets/kronodroid_real_device-balanced.csv'],
        "dense_layer_sizes_g": ['4096'],
        "dense_layer_sizes_d": ['2048'],
        'number_epochs': ['500'],
        'k_fold': ['10'],
        "num_samples_class_benign": ['10000'],
        "num_samples_class_malware": ['10000'],
        'training_algorithm': ['Adam'],
    },
    'kronodroid_r_jbcs_2025': {
        'input_dataset': ['datasets/reduced_balanced_kronodroid_real_device.csv'],
        "dense_layer_sizes_g": ['512'],
        "dense_layer_sizes_d": ['256'],
        'number_epochs': ['5000'],
        'k_fold': ['10'],
        'dropout_decay_rate_d': ['0.05'],
        'dropout_decay_rate_g': ['0.025'],
        'initializer_deviation': ['0.4'],
    },
    'kronodroid_e_jbcs_2025': {
        'input_dataset': ['datasets/reduced_balanced_kronodroid_emulator.csv'],       
        "dense_layer_sizes_g": ['512'],  
        "dense_layer_sizes_d": ['256'],
        'number_epochs': ['5000'],
        'k_fold': ['10'], 
        'dropout_decay_rate_d': ['0.05'],
        'dropout_decay_rate_g': ['0.025'],
        'initializer_deviation': ['0.4'],
    },
    'android_p_jbcs_2025': {
       'input_dataset': ['datasets/reduced_balanced_android_permissions.csv'],
       "dense_layer_sizes_g": ['1024'],
       "dense_layer_sizes_d": ['512'],
       'number_epochs': ['1000'],
       'k_fold': ['10'],
       'dropout_decay_rate_d': ['0.4'],
       'dropout_decay_rate_g': ['0.2'],
       'initializer_deviation': ['0.5'],
     },
     'adroit_jbcs_2025': {
        'input_dataset': ['datasets/reduced_balanced_adroit.csv'],
        "dense_layer_sizes_g": ['64'],
        "dense_layer_sizes_d": ['32'],
        'number_epochs': ['5000'],
        'k_fold': ['10'],
        'dropout_decay_rate_d' : ['0.1'],
        'dropout_decay_rate_g': ['0.05'],
        'initializer_deviation': ['0.5'],
    },
     'drebin_jbcs_2025': {
        'input_dataset': ['datasets/reduced_balanced_drebin215.csv'],
        "dense_layer_sizes_g": ['2048'],
        "dense_layer_sizes_d": ['1024'],
        'number_epochs': ['5000'],
        'k_fold': ['10'],
        'dropout_decay_rate_d' : ['0.4'],
        'dropout_decay_rate_g': ['0.2'],
        'initializer_deviation': ['0.5'],
    },
     'androcrawl_jbcs_2025': {
        'input_dataset': ['datasets/reduced_balanced_androcrawl.csv'],
        "dense_layer_sizes_g": ['2048'],
        "dense_layer_sizes_d": ['512'],
        'number_epochs': ['2000'],
        'k_fold': ['10'],
        'dropout_decay_rate_d' : ['0.4'],
        'dropout_decay_rate_g': ['0.2'],
        'initializer_deviation': ['0.5'],
      },
}

def print_config(Parametros):
    """
    Imprime a configuração dos argumentos para fins de logging.

    Parametros:
        Parametros : Argumentos de linha de comando.
    """
    logging.info("Command:\n\t{0}\n".format(" ".join([x for x in sys.argv])))
    logging.info("Settings:")
    lengths = [len(x) for x in vars(Parametros).keys()]
    max_length = max(lengths)

    for k, v in sorted(vars(Parametros).items()):
        message = "\t" + k.ljust(max_length, " ") + " : {}".format(v)
        logging.info(message)
    logging.info("")

def convert_flot_to_int(value):
    """
    Converte um valor float para int multiplicando por 100.

    Parametros:
        value: Valor a ser convertido.

    Retorno:
        value: Valor convertido.
    """
    if isinstance(value, float):
        value = int(value * 100)
    return value

class IntRange:
    """
    Tipo personalizado de argparse que representa um inteiro delimitado por um intervalo.

    Funções:
        - __init__: Inicializa a classe com os limites inferior e superior opcionais.
        - __call__: Converte o argumento fornecido para inteiro e verifica se está dentro do intervalo.
        - exception : Retorna uma exceção ArgumentTypeError com uma mensagem de erro apropriada.
    """

    def __init__(self, imin=None, imax=None):
        """
        Inicializa a classe IntRange com limites opcionais.

        Parametros:
            imin : Limite inferior do intervalo. O valor padrão é None.
            imax : Limite superior do intervalo. O valor padrão é None.
        """
        self.imin = imin
        self.imax = imax

    def __call__(self, arg):
        """
        Converte o argumento fornecido para inteiro e verifica se está dentro do intervalo especificado.

        Parametros:
            arg : O argumento fornecido na linha de comando.

        Retorno:
            int: O valor convertido se estiver dentro do intervalo.

        Exceções:
            ArgumentTypeError: Se o argumento não puder ser convertido para inteiro ou não estiver dentro do intervalo.
        """
        try:
            value = int(arg)
        except ValueError:
            raise self.exception()

        if (self.imin is not None and value < self.imin) or (self.imax is not None and value > self.imax):
            raise self.exception()

        return value

    def exception(self):
        """
        Retorna uma exceção ArgumentTypeError com uma mensagem de erro apropriada.

        Retorno:
            ArgumentTypeError: Exceção com uma mensagem que especifica os limites do intervalo.
        """
        if self.imin is not None and self.imax is not None:
            return argparse.ArgumentTypeError(f"Must be an integer in the range [{self.imin}, {self.imax}]")
        elif self.imin is not None:
            return argparse.ArgumentTypeError(f"Must be an integer >= {self.imin}")
        elif self.imax is not None:
            return argparse.ArgumentTypeError(f"Must be an integer <= {self.imax}")
        else:
            return argparse.ArgumentTypeError("Must be an integer")
def run_cmd(cmd, shell=False):
    """
    A função executa um comando de shell especificado e registra a saída.

    Parametros:
        cmd : Comando a ser executado.
        shell : Indica se deve usar o shell para executar o comando.
    """
    logging.info("Command line  : {}".format(cmd))
    cmd_array = shlex.split(cmd)
    logging.debug("Command array: {}".format(cmd_array))
    if not Parametros.demo:
        subprocess.run(cmd_array, check=True, shell=shell)

class Campaign:
    """
    Classe que representa uma campanha de treino.
    """
    def __init__(self, datasets, training_algorithm, dense_layer_sizes_g, dense_layer_sizes_d):
        self.datasets = datasets
        self.training_algorithm = training_algorithm
        self.dense_layer_sizes_g = dense_layer_sizes_g
        self.dense_layer_sizes_d = dense_layer_sizes_d

def check_files(files, error=False):
    """
    Verifica se os arquivos especificados existem.

    Parametros:
        files: Arquivos a verificar.
        error: Indica se deve lançar erro se o arquivo não for encontrado.

    Retorno:
        bool: True se todos os arquivos forem encontrados, False caso contrário.
    """
    internal_files = files if isinstance(files, list) else [files]

    for f in internal_files:
        if not os.path.isfile(f):
            if error:
                logging.info("ERROR: file not found! {}".format(f))
                sys.exit(1)
            else:
                logging.info("File not found! {}".format(f))
                return False
        else:
            logging.info("File found: {}".format(f))
    return True

def main():
    """
    Função principal que configura e executa as campanhas.
    """

    parser = argparse.ArgumentParser(description='Torrent Trace Correct - Machine Learning')
    #definição dos arugmentos de entrada
    help_msg = "Campaign {} (default={})".format([x for x in campaigns_available.keys()], DEFAULT_CAMPAIGN)
    parser.add_argument("--campaign", "-c", help=help_msg, default=DEFAULT_CAMPAIGN, type=str)
    parser.add_argument("--demo", "-d", help="demo mode (default=False)", action='store_true')
    help_msg = "verbosity logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
    parser.add_argument("--verbosity", "-v", help=help_msg, default=DEFAULT_VERBOSITY_LEVEL, type=int)
    parser.add_argument('-ml','--use_mlflow',action='store_true',help="Uso ou não da ferramenta mlflow para monitoramento") 

    parser.add_argument("--dense_layer_sizes_g", type=list_of_ints,default=None,help=" Valor das camadas densas do gerador")
    parser.add_argument("--dense_layer_sizes_d", type=list_of_ints,default=None,help="valor das camadas densas do discriminador")
    parser.add_argument('--number_epochs', type=list_of_ints,help='Número de épocas (iterações de treinamento).')
    parser.add_argument('--batch_size', type=int,default=256,choices=[16, 32, 64,128,256],help='Tamanho do lote da cGAN.')
    parser.add_argument("--optimizer_generator_learning", type=list_of_floats,default=None,help='Taxa de aprendizado do gerador')
    parser.add_argument("--optimizer_discriminator_learning", type=list_of_floats,default=None,help='Taxa de aprendizado do discriminador')

    parser.add_argument("--dropout_decay_rate_d",type=list_of_floats,default=None,help="Taxa de decaimento do dropout do discriminador da cGAN")
    parser.add_argument("--dropout_decay_rate_g",type=list_of_floats,default=None,help="Taxa de decaimento do dropout do gerador da cGAN")

    parser.add_argument('--initializer_mean', type=list_of_floats,default=None,help='Valor central da distribuição gaussiana do inicializador')
    parser.add_argument('--initializer_deviation', type=list_of_floats,default=None,help='Desvio padrão da distribuição gaussiana do inicializador')
    parser.add_argument('-p_ml','--port_mlflow',type=int,help="porta para o servidor mlflow",default=6002) 
    

    global Parametros
    Parametros = parser.parse_args()
    #cria a estrutura dos diretórios de saída
    print("Creating the structure of directories...")
    for p in PATHS:
        Path(p).mkdir(parents=True, exist_ok=True)
    print("done.\n")
    output_dir = 'outputs/out_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging_filename = '{}/evaluation_campaigns.log'.format(output_dir)

    logging_format = '%(asctime)s\t***\t%(message)s'
    if Parametros.verbosity == logging.DEBUG:
        logging_format = '%(asctime)s\t***\t%(levelname)s {%(module)s} [%(funcName)s] %(message)s'
    logging.basicConfig(format=logging_format, level=Parametros.verbosity)

    rotatingFileHandler = RotatingFileHandler(filename=logging_filename, maxBytes=100000, backupCount=5)
    rotatingFileHandler.setLevel(Parametros.verbosity)
    rotatingFileHandler.setFormatter(logging.Formatter(logging_format))
    logging.getLogger().addHandler(rotatingFileHandler)

    print_config(Parametros)
    # Tratamento das campanhas escolhidas
    campaigns_chosen = []
    if Parametros.campaign is None:
        campaigns_chosen = campaigns_available.keys()
    else:
        if Parametros.campaign in campaigns_available.keys():
            campaigns_chosen.append(Parametros.campaign)
        elif ',' in Parametros.campaign:
            campaigns_chosen = Parametros.campaign.split(',')
        else:
            logging.error("Campaign '{}' not found".format(Parametros.campaign))
            sys.exit(-1)
    # Obtém o tempo de início da execução
    time_start_campaign = datetime.datetime.now()
    logging.info("\n\n\n")
    logging.info("##########################################")
    logging.info(" EVALUATION ")
    logging.info("##########################################")
    time_start_evaluation = datetime.datetime.now()
    count_campaign = 1
    aux=None

    USE_MLFLOW=False
    #testa se o parametro do mlflow está ativado
    if Parametros.use_mlflow:
         USE_MLFLOW= True
    if USE_MLFLOW==False:
        for c in campaigns_chosen:
            #inicialização a execuçao sem mflow

                logging.info("\tCampaign {} {}/{} ".format(c, count_campaign, len(campaigns_chosen)))
                #para cada campanha aumentar o número de campanhas
                #count_campaign += 1
                campaign = campaigns_available[c]
                if(Parametros.dense_layer_sizes_g!=None):
                    campaign['dense_layer_sizes_g']=Parametros.dense_layer_sizes_g
                if(Parametros.dense_layer_sizes_d!=None):
                    campaign['dense_layer_sizes_d']=Parametros.dense_layer_sizes_d
                if(Parametros.number_epochs!=None):
                    campaign['number_epochs']=Parametros.number_epochs
                if(Parametros.optimizer_generator_learning!=None):
                    campaign['optimizer_generator_learning']=Parametros.optimizer_generator_learning
                if(Parametros.optimizer_discriminator_learning!=None):
                    campaign["optimizer_discriminator_learning"]=Parametros.optimizer_discriminator_learning
                if(Parametros.dropout_decay_rate_d!=None):
                    campaign["dropout_decay_rate_d"]=Parametros.dropout_decay_rate_d
                if(Parametros.dropout_decay_rate_g!=None):
                    campaign["dropout_decay_rate_g"]=Parametros.dropout_decay_rate_g
                if(Parametros.initializer_mean!=None):
                    campaign["initializer_mean"]=Parametros.initializer_mean
                if(Parametros.initializer_deviation!=None):
                    campaign['initializer_deviation']=Parametros.initializer_deviation
                params, values = zip(*campaign.items())
                combinations_dicts = [dict(zip(params, v)) for v in itertools.product(*values)]
                #print(campaign["output_dir"][0])

                #campaign_dir = '{}/{}'.format(output_dir, c)
                count_combination = 1
                for combination in combinations_dicts:
                    logging.info("\t\tcombination {}/{} ".format(count_combination, len(combinations_dicts)))
                    logging.info("\t\t{}".format(combination))
                    # estabelece o comando de execução
                    cmd = COMMAND
                    cmd += " --verbosity {}".format(Parametros.verbosity)
                    cmd+=" --batch_size {}".format(Parametros.batch_size)
                    #count_combination += 1
                    count_combination=1
                    for param in combination.keys():
                        cmd += " --{} {}".format(param, combination[param])
                        if(param=="input_dataset"):

                            cmd+=" --output_dir {}".format((c+"/"+((combination[param].split("/")[-1]).split('.csv')[0])+'_'+str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))))

                    count_combination += 1
                        
                    # cronometra o início do experimento da campanha
                    time_start_experiment = datetime.datetime.now()
                    logging.info("\t\t\t\t\tBegin: {}".format(time_start_experiment.strftime(TIME_FORMAT)))
                    run_cmd(cmd)
                    #cronometra o fim do experimento da campanha
                    time_end_experiment = datetime.datetime.now()
                    duration = time_end_experiment - time_start_experiment
                    logging.info("\t\t\t\t\tEnd                : {}".format(time_end_experiment.strftime(TIME_FORMAT)))
                    logging.info("\t\t\t\t\tExperiment duration: {}".format(duration))

                time_end_campaign = datetime.datetime.now()
                logging.info("\t Campaign duration: {}".format(time_end_campaign - time_start_campaign))
        #Obtém o tempo de final da execução
        time_end_evaluation = datetime.datetime.now()
        logging.info("Evaluation duration: {}".format(time_end_evaluation - time_start_evaluation))
    else:
        #caso o mlflow esteja habilitado, estabelece o endereço e nome da campanha
        adress="http://127.0.0.1:{}".format(Parametros.port_mlflow)
        mlflow.set_tracking_uri(adress)

          
        for c in campaigns_chosen:
            mlflow.set_experiment(c)
            #mlflow.set_experiment(c)

           #para cada execução da campanha é criada uma execução filha da execução original
            logging.info("\tCampaign {} {}/{} ".format(c, count_campaign, len(campaigns_chosen)))
            count_campaign += 1
            campaign = campaigns_available[c]
            if(Parametros.dense_layer_sizes_g!=None):
                    campaign['dense_layer_sizes_g']=Parametros.dense_layer_sizes_g
            if(Parametros.dense_layer_sizes_d!=None):
                    campaign['dense_layer_sizes_d']=Parametros.dense_layer_sizes_d
            if(Parametros.number_epochs!=None):
                    campaign['number_epochs']=Parametros.number_epochs
            if(Parametros.optimizer_generator_learning!=None):
                    campaign['optimizer_generator_learning']=Parametros.optimizer_generator_learning
            if(Parametros.optimizer_discriminator_learning!=None):
                    campaign["optimizer_discriminator_learning"]=Parametros.optimizer_discriminator_learning
            if(Parametros.dropout_decay_rate_d!=None):
                    campaign["dropout_decay_rate_d"]=Parametros.dropout_decay_rate_d
            if(Parametros.dropout_decay_rate_g!=None):
                    campaign["dropout_decay_rate_g"]=Parametros.dropout_decay_rate_g
            if(Parametros.initializer_mean!=None):
                    campaign["initializer_mean"]=Parametros.initializer_mean
            if(Parametros.initializer_deviation!=None):
                    campaign['initializer_deviation']=Parametros.initializer_deviation
            params, values = zip(*campaign.items())
            combinations_dicts = [dict(zip(params, v)) for v in itertools.product(*values)]
            campaign_dir = '{}/{}'.format(output_dir, c)
            count_combination = 1
            for combination in combinations_dicts:
                run_name="{}".format((c+"/"+((combination["input_dataset"].split("/")[-1]).split('.csv')[0])+'_'+str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))))

                with mlflow.start_run(nested=True) as run:
                    id=run.info.run_id    
                    logging.info("\t\tcombination {}/{} ".format(count_combination, len(combinations_dicts)))
                    logging.info("\t\t{}".format(combination))
                    # estabelece o comando de execução
                    cmd = COMMAND2
                    cmd += " --verbosity {}".format(Parametros.verbosity)
                    cmd+=" --batch_size {}".format(Parametros.batch_size)
                    cmd += " --run_id {}".format(id)
                    #count_combination += 1
                    count_combination=1
                    for param in combination.keys():
                        cmd += " --{} {}".format(param, combination[param])
                        if(param=="input_dataset"):

                            cmd+=" --output_dir {}".format(run_name)
                    cmd+= " --port_mlflow {}".format(Parametros.port_mlflow)
                    count_combination += 1
                    # cronometra o início do experimento da campanha
                    time_start_experiment = datetime.datetime.now()
                    logging.info(
                        "\t\t\t\t\tBegin: {}".format(time_start_experiment.strftime(TIME_FORMAT)))
                    run_cmd(cmd)
                    #cronometra o fim do experimento da campanha
                    time_end_experiment = datetime.datetime.now()
                    duration = time_end_experiment - time_start_experiment
                    logging.info("\t\t\t\t\tEnd                : {}".format(time_end_experiment.strftime(TIME_FORMAT)))
                    logging.info("\t\t\t\t\tExperiment duration: {}".format(duration))


            time_end_campaign = datetime.datetime.now()
            logging.info("\t Campaign duration: {}".format(time_end_campaign - time_start_campaign))
        #Obtém o tempo de final da execução
        time_end_evaluation = datetime.datetime.now()
        logging.info("Evaluation duration: {}".format(time_end_evaluation - time_start_evaluation))



if __name__ == '__main__':
    sys.exit(main())

