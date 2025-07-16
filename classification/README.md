# Classification

This repository contains the code for training and testing different BERT-based models for TTP extraction from text.

## 1. Initial setup

This code is tested and developed on a machine running `Ubuntu 22.04.4 LTS (jammy)`, `Python 3.10.12`, `CUDA 12.4`, with 2 NVIDIA L40. 
For full compatibility, this project requires a GPU having similar VRAM size. 

First, clone this repository and move to the project folder. 
Make sure that in the project folder you have **non-empty** folders `fine_tuned` (contains BERT-based fine-tuned models), `configs/` (contains unlabeled classifier thresholds), `datasets` (contains training and test datasets), and `local`. They can be found in the Zenodo repository, in `tgz` format.

For reference, the (partial!) directory tree should look as follows:
```
.
├── configs
│   ├── all_sentence_similarity.json
│   └── open_classification.json
├── datasets
│   ├── bosch_test.json
│   ├── bosch_train.json
│   ├── converter.py
│   ├── enterprise-attack.json
│   ├── mitre_embeddings.pickle
│   ├── nvidia-bosch-test-embeddings.pickle
│   ├── nvidia-bosch-train-embeddings.pickle
│   ├── nvidia-mitre-embeddings.pickle
│   ├── nvidia-tram-test-embeddings.pickle
│   ├── nvidia-tram-train-embeddings.pickle
│   ├── ood_data.csv
│   ├── syn_sentence_list.json
│   ├── tram_test.json
│   ├── tram_train_augmented_artificial.json
│   ├── tram_train_augmented_ood.json
│   └── tram_train.json
├── fine_tuned  # pre-trained BERT-models
│   ├── bosch_swipe  # should contain more than 10 models
│   ├── data_augmentation  # contains exactly 2 models
│   └── tram_swipe  # contains more than 10 models
├── local
│   ├── CyBERT-Base-MLM-v1.1  # original CyBERT model
│   ├── scibert_multi_label_model  # original TRAM model
│   └── tram_finetuned  
.
.

!  continues!

├── README.md
.
.
```

> [!IMPORTANT]  
> [DarkBERT](https://arxiv.org/abs/2305.08596) requires an access token to be used, and can be asked to the authors of the work. Place in `.darkbert_token` your token for accessing the model `s2w-ai/DarkBERT`. This must step should be performed before building the Dockerfile.

### 1.1 Build
Make sure you have `docker` installed on your system.
To build the project, you need to execute the following command:
```bash
$ docker build --build-arg RUN_DEVICE=YOUR_DEVICE -t sok-classification .
```

The parameter `RUN_DEVICE` makes you select the GPU device where to run the project (e.g., `cuda`, `cuda:0`, `cpu`, etc.). By default, this parameter is set to `cuda`. This, will not work if your system doesn't have a GPU (strongly recommended for this project).

You can test the successful build by executing the following:

```bash
$ docker run --gpus all -it sok-classification
```

This should partially reproduce the results shown in Table 6 of our paper.

## 2. Usage

If you want to use the project, you can open the container with:
```
$ docker run --gpus all -it sok-classification bash
```

Now, for calculating the results of Tables 6, 7, 8, and Figure 4 of the original paper -- which all use classification models -- you can execute the bash scripts contained in `artifact_eval/`.
For example:

```bash
$ ./artifact_eval/gen_table_6_annoctr.sh
```

The scripts contained in the `artifact_eval/` folder will use under the hood our fine-tuned models (`fine_tuned/` and `configs/`).


## 3. Labeled sentence classifiers
### 3.1 Training
#### 3.1.1 Setup Grid and Generated Config Files

Specify the parameters in `grid.py`. Each parameter of the grid will be combined to generate the different configurations.

You can generate the configuration files with:
```bash
$ python gen_json_config.py multi OUTFILE [ID_NUMBER]
```
The results will be placed in `OUTFILE`, (e.g, `multi_label_setup.json`). Each configuration will have an "id" number, specified through the optional `ID` parameter. If not set, this will be automatically set to 0. You can inspect the file and delete setups that you wish to ignore.

#### 3.1.2 Start Training

Train the models listed in the setup with:
```bash
$ python train_multilabel.py CONFIG_FILE DEVICE IDX_START N_READ
```
The parameters have the following purpose:
* `CONFIG_FILE` is the file generated with `gen_json_config.py` (see Section 2.1.1 of this document);
* `DEVICE` is the device where tensors and models will be loaded (e.g., `cuda`, `cpu`);
* `IDX_START` represents the index of the config file where to start reading configurations;
* `N_READ` is the number of configurations that will be trained (i.e., from `IDX_START` to `IDX_START`+`N_READ`)

Results for each configuration will show up in `results/multi_label/{CONF_ID}_{ÐATASET_NAME}_{MODEL_NAME}`. The folder will contain the following elements:  
- `loss.pdf`: nice plot with training and validation loss at each epoch;  
- `model_chkp`: checkpoint of model with lowest validation loss;
- `stats.json`: summary of model performances over time;  
- `classification_reports.txt`: list of validation classification reports at each training epoch;  
- `model_params.json`: parameters used by the training script when training the model  

All the other files will be used to load the pretrained model.

> [!IMPORTANT]
> Please note that some models (e.g., RoBERTa-Large, XLM-RoBERTa-Large) may require higher "patience" (specified in `train_common.py`). In some cases, training terminates earlier due to the loss function reaching a local minimum, and additional training steps are required to overcome it. You can fix the parameter directly from `train_common.py`, or specify it with the environment variable `PATIENCE`.

### 3.2 Test

To test the trained models, run as module `test_labeled.py`. Alternatively, you can also use the `test_supervised.ipynb` notebook. The python script contains command arguments that more easily allow the execution. Both have the same functionalities.
The script will ask you to provide an input folder that contains all the trained models and automatically tests them with the corresponding datasets (model trained with `bosch_t10` will use `bosch_t10` as test set). 

You can see here the help message from the `test_labeled` module:
```
$ python -m test_labeled -h
usage: test_labeled [-h] [--show-baseline] [--remove-dupl-models] model_dir outfile device

This code tests supervised BERT models stored inside a folder on their corresponding datasets.

positional arguments:
  model_dir             The base directory containing the models to test (e.g. "fine_tuned/tram_swipe").
  outfile               The output file to save the results.
  device                The device to use for testing (e.g., 'cuda:0' or 'cpu').

options:
  -h, --help            show this help message and exit
  --show-baseline       This flag will show the baseline results for the TRAM dataset.
  --remove-dupl-models  This flag will remove duplicate models from the results (e.g., two RoBERTa trained with different hyper-parameters)
```

In the notebook, you can specify by hand the fine-tuned models you want to load (e.g., `./fine_tuned/multi_label/0_tram_FacebookAI-roberta-large`).

## 4. Unlabeled sentence classifiers

### 4.1 Generate Embeddings
First, you need to generate the embeddings of all MITRE ATT&CK titles and descriptions. To do so, run:

```bash
$ python gen_mitre_embeddings.py
```

The embeddings will be placed in a python pickle in `datasets/mitre_embeddings.pickle`.

### 4.2 Hyperparameter Tuning and Testing

Run (as module) the script `unlabeled.py`. The script will make you choose two modes: `tuning`, `test`.
Alternatively, you can open `test_sentence_similarity.ipynb` and run notebook cells with the desired configurations. This notebook will find the optimal thresholds for the model and use it on the test set.
You can also use the script `unlabeled.py`. For testing *NVidia-embed*, refer to `test_sentence_similarity_gLLM.ipynb`, `unlabeled_nvidia_embed.py` (same of the notebook), and the scripts that generate the embeddings of our datasets, contained in the folder `./nvidia-embed/`.

## 5. Data Augmentation

First, you need to execute the `make_augmented_sets.ipynb` notebook.
Then, with the generated dataset, you need to train the supervised models on the augmented datasets by following the instruction of Section 3 of the present document. Training will automatically use the augmented dataset, while the test on the regular non-augmented dataset.

## 6. Models

All fine-tuned models are available in the folder `./fine_tuned/`. The folders are organized as follows:
* `tram_swipe`: Contains models fine-tuned on the TRAM2 dataset.
* `bosch_swipe`: Contains models fine-tuned on the AnnoCTR dataset.
* `data_augmentation`: Contains models fine-tuned on augmented datasets (OOD, Syntethic).

Other models, such as "TRAM2", and "CyBERT" can be found in the directory `./local/`. "CyBERT" is located in `CyBERT-Base-MLM-v1.1`. TRAM2 can be found in `scibert_multi_label_model` (original model), and `tram_finetuned` (original model without classification head). The latter is used for testing on the AnnoCTR dataset.

As specified in Section 1, the "DarkBERT" model requires a secret token for access. Therefore, we recommend the reader to require an access token to the original authors of the work.

