# BootBOGS 

## Overview

BootBOGS is an innovative approach that combines Bootstrap resampling and HyperOpt TPE (Tree-structured Parzen Estimator) to efficiently obtain optimal hyperparameters for machine learning models. This method significantly reduces the time required compared to traditional Grid Search. In this repository, we provide implementations of BootBOGS alongside other hyperparameter tuning methods for four different datasets: Diabetes, German Credit, Taiwan Bankruptcy, and Polish Bankruptcy (third year) datasets.

## Repository Contents

This repository contains the following files and directories:

- **diabetes_test.py**: This file contains the implementation for hyperparameter tuning using BootBOGS on the Diabetes dataset.
- **german_test.py**: This file contains the implementation for hyperparameter tuning using BootBOGS on the German Credit dataset.
- **taiwan_test.py**: This file contains the implementation for hyperparameter tuning using BootBOGS on the Taiwan Bankruptcy dataset.
- **bankruptcy_test.py**: This file contains the implementation for hyperparameter tuning using BootBOGS on the Polish Bankruptcy (third year) dataset.
- **datasets**: This directory contains the dataset files used for the experiments.
- **config**: This directory contains configuration file for optimization methods.
- **requirements.txt**: A list of Python dependencies required to run the code.
- **README.md**: The file you are currently reading, providing instructions and information about the repository.

## Usage

To run the hyperparameter tuning process for any of the datasets using BootBOGS, follow these steps:

1. Ensure you have Python installed on your system.

2. Install the required dependencies by running the following command in your terminal:

   ```bash
   pip install -r requirements.txt

3. Run the respective dataset script with the following command line arguments:

    ```bash
   python diabetes_test.py --dataset diabetes --numbers 10 --metric auc --path diabetes_rule01_config.yaml

Replace `diabetes_test.py` with the appropriate dataset file (e.g., `german_test.py`, `taiwan_test.py`, or `bankruptcy_test.py`) and adjust the arguments accordingly.

- `--dataset`: The name of the dataset you want to use.
- `--numbers`: The number of iterations for the hyperparameter search.
- `--metric`: The evaluation metric you want to optimize (e.g., 'auc').
- `--path`: The path to the configuration file for HyperOpt.

In our experiments, we use AUC for imbalanced datasets.

Replace `diabetes_test.py` with the appropriate dataset file (e.g., `german_test.py`, `taiwan_test.py`, or `bankruptcy_test.py`) and adjust the arguments accordingly.

- `--dataset`: The name of the dataset you want to use.
- `--numbers`: The number of iterations for the hyperparameter search.
- `--metric`: The evaluation metric you want to optimize (e.g., 'auc').
- `--path`: The path to the configuration file for HyperOpt.

## Methodology

BootBOGS combines Bootstrap resampling to create multiple subsets of the dataset and HyperOpt TPE to efficiently search for the best hyperparameters. The result is a set of optimal hyperparameters that can significantly reduce the time required for model tuning.

## Questions or Issues

If you have any questions or encounter any issues while using this repository, please feel free to [create an issue](https://github.com/thiphan94/BootBOGS/issues) on GitHub

