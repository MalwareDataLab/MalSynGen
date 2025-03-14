# MalSynGen: redes neurais artificiais na geração de dados tabulares sintéticos para detecção de malware
## MalSynGen
MalSynGen is a tool that uses artificial neural networks to generate synthetic tabular data for the Android malware domain. To evaluate evaluated the effectiveness of this approach using various datasets and metrics that assess the fidelity of the generated data, its utility in classification, and the computational efficiency of the process.  The results indicate that MalSynGen is capable of capturing representative patterns for tabular data augmentation.
### important information
This is a public tool, however, if you use this material, please reference the articles:
>@inproceedings{sbseg_estendido,<br/>
 author = {Angelo Nogueira and Kayua Paim and Hendrio Bragança and Rodrigo Mansilha and Diego Kreutz},<br/>
 title = { MalSynGen: redes neurais artificiais na geração de dados tabulares sintéticos para detecção de malware},<br/>
 booktitle = {Anais Estendidos do XXIV Simpósio Brasileiro de Segurança da Informação e de Sistemas Computacionais},<br/>
 location = {São José dos Campos/SP},<br/>
 year = {2024},<br/>
 keywords = {},<br/>
 issn = {0000-0000},<br/>
 pages = {129--136},<br/>
 publisher = {SBC},<br/>
 address = {Porto Alegre, RS, Brasil},<br/>
 doi = {10.5753/sbseg_estendido.2024.243359},<br/>
 url = {https://sol.sbc.org.br/index.php/sbseg_estendido/article/view/30126}}<br/>

 
https://sol.sbc.org.br/index.php/sbseg_estendido/article/view/30126

>@inproceedings{sbseg,<br/>
 author = {Angelo Nogueira and Kayua Paim and Hendrio Bragança and Rodrigo Mansilha and Diego Kreutz},<br/>
 title = { Geração de dados sintéticos tabulares para detecção de malware Android: um estudo de caso},<br/>
 booktitle = {Anais do XXIV Simpósio Brasileiro de Segurança da Informação e de Sistemas Computacionais},<br/>
 location = {São José dos Campos/SP},<br/>
 year = {2024},<br/>
 keywords = {},<br/>
 issn = {0000-0000},<br/>
 pages = {808--814},<br/>
 publisher = {SBC},<br/>
 address = {Porto Alegre, RS, Brasil},<br/>
 doi = {10.5753/sbseg.2024.241731},<br/>
 url = {https://sol.sbc.org.br/index.php/sbseg/article/view/30072}}<br/>


https://sol.sbc.org.br/index.php/sbseg/article/view/30072
## 1. Preparation and installation

1. Clone the repository and run the following commands.
```bash
git clone https://github.com/MalwareDataLab/MalSynGen.git
cd MalSynGen
```
2. Install Pipenv, which is necessary to run several commands.
```bash
pip install pipenv
```

3. Install the dependencies.

**Option 1**: Build a Docker image locally from the Dockerfile.

```bash
./scripts/docker_build.sh
```
**Option 2**: Use the **pip_env_install.sh** script.

```bash
./pip_env_install.sh
```

**Option 3**: Configure venv.
```
   python3 -m venv .venv
   ```
   ```
   source .venv/bin/activate
   ```
   ```
   pip3 install -r requirements.txt
   ```

   **Option 4**: Configure pipenv.

   ```
   pipenv install -r requirements.txt
   ```

## 2. Executiom
Run the demo of the tool: 

**Option 1**: Install the dependencies and run a demo in a Linux environment. The execution takes about 5 minutes on an AMD Ryzen 7 5800x, 8 cores, 64 GB RAM machine. 
```bash
./run_demo_venv.sh
```

**Option 2**: Run the docker demo that instantiates a reduced version of the experiment. The execution takes about 5 minutes on an AMD Ryzen 7 5800x, 8 cores, 64 GB RAM machine. 
```bash
./run_demo_docker.sh
```




### 2.1. Dependencies
We tested the tool code with the following Python versions:


1. Python 3.8.2 

2. Python 3.8.10

3. Python 3.9.2

4. Python 3.10.12

The MalSynGen code has dependencies on several Python packages and libraries, such as numpy 1.21.5, Keras 2.9.0, Tensorflow 2.9.1, pandas 1.4.4, scikit-learn 1.1.1. and mlflow 2.12.1. The complete and extensive list of dependencies is in the [**requirements.txt**](https://github.com/MalwareDataLab/MalSynGen/blob/171bd57290467a85566f3a5193a2f540ac44a2d9/requirements.txt) file.


## 3. Reproduction
To reproduce the same experiments (campaigns) as in the paper, use one of the following options. The execution takes around 14 hours on an AMD Ryzen 7 5800x, 8 cores, 64 GB RAM computer.

   **Opção 1**: On the local enviroment
   ```bash
   ./run_reproduce_sf24_venv.sh
   ```

 
   **Opção 2**:  On the docker enviroment
   ```bash
   ./run_reproduce_sf24_docker.sh
   ```
 
## 4. Other execution options
The **run_balanced_datasets.sh** script is responsible for executing the balanced datasets of the experiments based on the input specified by the user.
Run the script:


   ```bash
   ./run_balanced_datasets.sh
   ```


#### 4.1. Running other experiments

The tool relies on **run_campaign.py** to automate the training and evaluation of cGAN. **run_campaign.py** allows you to run multiple evaluation campaigns with different parameters, recording the results in output files for later analysis. The user will be able to visually perform a comparative analysis of the different configurations in relation to the datasets used.

Example of running a pre-configured campaign based on the Kronodroid R execution from the article:

```
pipenv run python3 run_campaign.py -c Kronodroid_r
```


Same campaign (Kronodroid_r) running directly in the application (**main.py**):


```
pipenv run python main.py --verbosity 20 --input_dataset datasets/kronodroid_real_device-balanced.csv --dense_layer_sizes_g 4096 --dense_layer_sizes_d 2048 --number_epochs 500 --k_fold 10 --num_samples_class_benign 10000 --num_samples_class_malware 10000 --training_algorithm Adam
```


#### 4.2. Automated test parameters:

      --------------------------------------------------------------

    --campaign ou -c:    Especifica a campanha de avaliação que você deseja executar. 
                         Você pode fornecer o nome de uma campanha específica ou uma  
                         lista de campanhas separadas por vírgula. 
                         Por exemplo: --campaign SF24_4096_2048_10 ou --campaign 
                          Kronodroid_e,kronodroid_r.

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


#### 4.3. Running the tool in Google Colab
Google collab is a cloud tool that allows you to run Python code in your browser.

1. Access the following link to use the Google colab tool: https://colab.google/

2. Create a new notebook by clicking the **New notebook** button at the top right of the screen.

<td><img src= https://github.com/SBSegSF24/MalSynGen/assets/174879323/628010a5-f2e9-48dc-8044-178f7e9c2c37 style="max-width:100%;"></td>

3. Upload the MalSynGen folder to your Google Drive.

4. Add a new cell by clicking the **+code** button at the top left of the screen, containing the following code snippet to access the Google Drive folder.


```
from google.colab import drive
drive.mount('/content/drive')
```
5. Create a cell to access the MalSynGen folder (Example):

```
cd /content/drive/MyDrive/MalSynGen-main
```
6. Install the tool dependencies by creating a cell with the following code:
```
!pip install -r requirements.txt
```
7. Create a cell to run the tool (Example):
```
!python main.py --verbosity 20 --input_dataset datasets/kronodroid_real_device-balanced.csv --dense_layer_sizes_g 4096 --dense_layer_sizes_d 2048 --number_epochs 500 --k_fold 10 --num_samples_class_benign 10000 --num_samples_class_malware 10000 --training_algorithm Adam
```


## 5. Execution flux
![Screenshot from 2024-07-05 17-00-39](https://github.com/SBSegSF24/MalSynGen/assets/174879323/4d55117e-4203-4930-a0f5-2ed19c8857e3)

The tool's execution flow consists of three steps:

**Dataset selection**: In this step, we perform balancing by the minority class, through the use of subsampling techniques. The balanced datasets and the code used in this process can be found at: https://github.com/SBSegSF24/MalSynGen/tree/accbe69f12bbf02d5d7f9c75291a60a5738bbb67/datasets

The balanced dataset is then processed in the training and evaluation steps, through k-fold cross-validation, where two subsets are created: evaluation subset (Dataset r) and training subset (Dataset R).

**Training**: In this step, cGANs are trained and used to generate synthetic data. We need to train classifiers to later verify the usefulness of the generated synthetic data: Dataset S (generated from R) and Dataset s (generated from r). The classifiers used are called TRTS (Trained with dataset R, evaluated with s) and TSTR (Trained with S, evaluated with r).

**Evaluation**:n the evaluation stage, we execute the classifiers and compute the primary metrics, utility and fidelity.






## 6.Tool Parameters
|       Flag/ parameter       |                                 Description                              | Required |
|:---------------------------:|:------------------------------------------------------------------------:|:--------:|
|     -i , --input_dataset    |                         Path to the input dataset                        |    yes   |
|       -o, --output_dir      |                    Directory for saving output files.                    |    no    |
|         --data_type         |              Data type to represent sample characteristics.              |    no    |
| --num_samples_class_malware |                  Number of Class 1 (malignant) samples.                  |    yes   |
|  --num_samples_class_benign |                    Number of Class 0 (benign) samples.                   |    yes   |
|       --number_epochs       |           Number of epochs (training interaction) for the cGAN.          |    no    |
|           --k_fold          |                    Number of cross-validation k folds                    |    no    |
|      --initializer_mean     |         Central value of the initializer's Gaussian distribution.        |    no    |
|   --initializer_deviation   |      Standard deviation of the initializer's Gaussian distribution.      |    no    |
|      --latent_dimension     |                 Latent space dimension for cGAN training.                |    no    |
|     --training_algorithm    | Training algorithm for the cGAN. Options: 'Adam', 'RMSprop', 'Adadelta'. |    no    |
|    --activation_function    |      cGAN activation function. Options: 'LeakyReLU', 'ReLU', 'PReLU'     |    no    |
|    --dropout_decay_rate_g   |                    cGAN generator dropout decay rate.                    |    no    |
|    --dropout_decay_rate_d   |                   cGAN discriminator dropout decay rate.                 |    no    |
|    --dense_layer_sizes_g    |               Values of the dense layers of the generator.               |    no    |
|    --dense_layer_sizes_d    |             Values of the dense layers of the discriminator.             |    no    |
| --latent_mean_distribution" |              Mean of the distribution of random input noise              |    no    |
| --latent_stander_deviation" |                 Standard deviation of random input noise                 |    no    |
|         --batch_size        |               cGAN batch size. Opções: 16, 32, 64, 128, 256              |    no    |
|         --verbosity         |                             Verbosity level;                             |    no    |
|        --save_models        |                       Option for saving the models.                      |    no    |
|   --path_confusion_matrix   |                  Output directory of confusion matrices.                 |    no    |
|      --path_curve_loss      |                Output directory for training curve graphs.               |    no    |
|        -a, --use_aim        |                   Option to use Aimstack tracking tool.                  |    no    |
|      -ml, --use_mlflow      |                  Option to use the mlflow tracking tool.                 |    no    |
|        -rid, --run_id       |      Option linked to mlflow, used to resume an unfinished execution     |    no    |
|    -tb, --use_tensorboard   |                  Option to use Tensorboard tracking tool                 |    no    |

## 7. Testing Environments

The tool has been successfully run and tested on the following environments:

1. **Hardware**: AMD Ryzen 7 5800x, 8 cores, 64 GB RAM. **Software**: Ubuntu Server 22.04.2 and 22.04.3, Python 3.8.10 and 3.10.12, Docker 24.07.

2. **Hardware**: Intel Core i7-9700 CPU 3.00GHz, 8 cores, 16 GB RAM. **Software**: Debian GNU 11 and 12, Python 3.9.2 and 3.11.2, Docker 20.10.5 and 24.07.

3. **Hardware**: AMD Ryzen 7 5800X 8-core, 64GB RAM (3200MHz), NVDIA RTX3090 24GB. **Software**: Python 3.11.5, WSL: 2.2.4.0, Docker version 24.0.7, 

## 8. Datasets
 The datasets used in this study were obtained from the Malware-Hunter project public repository, which can be found at https://github.com/MalwareDataLab/Datasetstree/44b14d78f1361a2300daa42b3d4127df8fad7068/JBCS_2025
 


## 9. Tracking tools
### 9.1. Aimstack

1. Install the tool:

```bash
pip install aim
```

2. Run MalSynGen with the -a or --use_aim option (Example):
```bash
pipenv run python3 main.py -i datasets/kronodroid_real_device-balanced.csv --num_samples_class_benign 10000 --num_samples_class_malware 10000 --batch_size 256 --number_epochs 300 --k_fold 10 -a
```
3. After the execution is complete, use the **aim up** command in the MalSynGen folder.
```bash
aim up
```
Aimstack Documentation: https://aimstack.readthedocs.io/en/latest/


### 9.2. Mlflow

1. Instalar a ferramenta:
   
```bash
pip install mlflow
```

2. Instanciar um servidor local na porta 6002:

```bash
mlflow server --port 6002
```
3. Executar MalSynGen com a opção -ml ou --use_mlflow (Exemplo):
```bash
pipenv run python3 main.py -i datasets/kronodroid_real_device-balanced.csv  --num_samples_class_benign 10000 --num_samples_class_malware 10000 --batch_size 256 --number_epochs 300 --k_fold 10 -ml
```

4. After the execution is complete, access the address http://localhost:6002/ in your browser to view the results.

Mlflow documentation: https://mlflow.org/docs/latest/index.html

### 9.3. Tensorboard

1. Install the tool:

```bash
pip install tensorboard
```

2. Run MalSynGen with the -tb or --use_tensorboard option (Example):
```bash
pipenv run python3 main.py -i datasets/kronodroid_real_device-balanced.csv --num_samples_class_benign 10000 --num_samples_class_malware 10000 --batch_size 256 --number_epochs 300 --k_fold 10 -tb
```

3. View the results with the command:

```bash
tensorboard --logdir=tensorboardfolder/ --port=6002
```


TensorBoard Documentation: https://www.tensorflow.org/tensorboard/get_started?hl=en
## 10. Code Documentation Overview
The code documentation is available in html format in the [docs](https://github.com/SBSegSF24/MalSynGen/tree/f89ddcd20f1dc4531bff671cc3a08a8d5e7c411d/docs) folder, to access the documentation open the [index.html](https://github.com/SBSegSF24/MalSynGen/blob/f89ddcd20f1dc4531bff671cc3a08a8d5e7c411d/docs/index.html) file in your local environment.

The main page contains the README.md documentation
![image](https://github.com/SBSegSF24/MalSynGen/assets/174879323/4a738e53-ebae-4e5b-99ad-9de269139cc7)

### 10.1. Related pages
![image](https://github.com/SBSegSF24/MalSynGen/assets/174879323/23ad1214-4644-49f7-ba60-a574575a8cc3)
The **Related pages** tab contains the appendix artifact information.
### 10.2. Namespace
![image](https://github.com/SBSegSF24/MalSynGen/assets/174879323/8fb63e75-ff4e-482f-855e-52e471fa90fb)
The **namespace** tab describes the modules and functions of the tool code.
### 10.3. Classes
![image](https://github.com/SBSegSF24/MalSynGen/assets/174879323/84f66aa6-ec28-4894-8c3a-e5d6e5f0226b)
The **Classes** tab contains the classes used, their hierarchies, variables and implementations in the tool implementation.


