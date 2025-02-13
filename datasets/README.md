# Datasets - Binaries - Balanced Datasets

## Overview

This dataset has been carefully balanced to include 10,000 samples per class (malware and benign) to ensure a fair and unbiased analysis. To optimize the dataset for efficient analysis and machine learning tasks, a maximum of 200 features have been selected using the Chi-Squared feature selection method. Additionally, irrelevant features, which have a single value across all samples, have been deleted to enhance the quality and usability of the data.


## Binaries

|             Dataset               | Features | Malwares | Benigns | Total |
|:---------------------------------:|:--------:|:--------:|:-------:|:-----:|
|             ADROIT                |   118    |   3418   |   3418  |  6836 |
|           AndroCrawl              |   136    |   10170  |  10170  | 20340 |
|       Android Permissions         |   148    |   9077   |   9077  | 18154 |
|  DefenseDroid APICalls Closeness  |   200    |   5222   |   5222  | 10444 |
|   DefenseDroid APICalls Degree    |   200    |   5222   |   5222  | 10444 |
|    DefenseDroid APICalls Katz     |   200    |   5222   |   5222  | 10444 |
|      DefenseDroid PRS[^PRS]       |   200    |   5975   |   5975  | 11950 |
|           DREBIN-215              |   200    |   5555   |   5555  | 11110 |
|      KronoDroid Real Device       |   200    |   10000  |  10000  | 20000 |
|        KronoDroid Emulator        |   200    |   10000  |  10000  | 20000 |

[^PRS]: Permissions, Receivers and Services

## Usage

### Downloading
- Clone or download this directory to your local machine.

### Accessing the Datasets
- The datasets are provided as CSV files. Use Python with pandas or any compatible tool to load and analyze the data.

### Analysis
- Utilize these binary-feature datasets for binary classification, clustering, and other malware-related research.

### Citation
- If you utilize these datasets in your research, please cite the repository or provide appropriate attribution to the original source.

## License
This dataset is released under the [CC BY 4.0 (Creative Commons Attribution 4.0 International)](https://creativecommons.org/licenses/by/4.0/).

## Feedback and Contributions
Feedback, suggestions, and contributions are encouraged! Contact us or submit a pull request if you wish to contribute.
