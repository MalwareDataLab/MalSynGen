## Artefatos apêndice SBSeg24/SF: #243359: MalSynGen: redes neurais artificiais na geração de dados tabulares sintéticos para detecção de malware

A MalSynGen  é uma ferramenta que utiliza redes neurais artificiais para gerar dados sintéticos tabulares para o domínio de malware Android. Para avaliar o seu desempenho foram aumentados os dados de dois datasets, considerando métricas de fidelidade estatística e utilidade. Os resultados indicam que MalSynGen é capaz de capturar padrões representativos para o aumento de dados tabulares.

## 1. Selos Considerados
Os autores solicitam a avaliação para os seguintes selos:

**SeloD - Artefatos Disponíveis**:
Artefatos disponíveis em repositório estável e público no GitHub, com documentação da ferramente, instruções de instalação e exemplos de utilização.

**SeloF - Artefatos Funcionais**:
 Artefatos funcionais e testados em Ubuntu 22.04 (*bare metal* e Docker), Ubuntu 20.04 (Docker), Debian  11 (*bare metal*) e Debian 12 (*bare metal*). Detalhes do hardware e software desses ambientes estão disponíveis no README.md do GitHub.

**SeloR - Artefatos Reprodutíveis**:
 Disponibilizamos um script para reprodução dos experimentos do trabalho. O script a informações sobre estimativa de execução estão disponíveis no README.md do GitHub.

**SeloS - Artefatos Sustentáveis**:
 Código estrutura, organizado, inteligível e acompanhado de boa documentação.


## 2. Informações básicas

As instruções de instalação, execução e utilização, bem como os códigos fonte, estão disponíveis no repositório GitHub https://github.com/SBSegSF24/MalSynGen. No repositório há um README.md contendo informações sobre o fluxo de execução da ferramenta, configuração, parâmetros de entrada e instalação nos seguintes ambientes:

1. *Bare metal* (testado em  Ubuntu 22.04 com Python 3.10.12,3.8.10 e 3.8.2 e em Debian 11 e 12 com Python 3.9.2)

2. *Google Colab* 

3. *Containers* Docker (testado em Docker versões 24.0.7 e 20.10.5 com imagem Ubuntu 20.04 e 22.04)


### 2.1. Dependências
Testamos o código da ferramenta com as seguintes versões Python:

1. Python 3.8.2 

2. Python 3.8.10

3. Python 3.9.2

4. Python 3.10.12

O código da MalSynGen possui dependências com diversos pacotes e bibliotecas Python, como 
numpy 1.21.5, Keras 2.9.0, Tensorflow 2.9.1, pandas 1.4.4, scikit-learn 1.1.1. e mlflow 2.12.1.
A lista completa e extensa das dependências está no arquivo **requirements.txt** do repositório GitHub. 


## 3. Instalação 

As instruções detalhadas de instalação em ambiente Linux local e/ou Docker estão disponíveis no README.md.

## 4. Datasets

O diretório **datasets** do GitHub contém os conjuntos de dados balanceados KronoDroid_emulator e KronoDroid_real_device utilizados nos experimentos do trabalho. O código utilizado para balancear os datasets originais também está disponível. O diretório **datasets** contém também os arquivos de validação de cada dataset e código de validação utilizado no subdiretório **validation**. As versões originais dos datasets tem como origem o repositório [https://github.com/aleguma/kronodroid](https://github.com/aleguma/kronodroid).



## 5. Ambiente de testes

A ferramenta foi testada em 3 ambientes de hardware distintos e mais de 5 ambientes de software distintos. Os detalhes estão disponíveis no README.md. 


## 6. Teste mínimo

As instruções de teste mínimo, entre outros, estão disponíveis README.md. Lá o usuário encontrará também estimativas de tempo de execução para cada teste.

## 7. Experimentos

Para reproduzir os experimentos do trabalho siga as intruções de reprodução do README.md. Os resultados serão salvos na pasta **campanhas_SF24**. 

