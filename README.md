# CogAware

CogAware is an emotion recognition neural network implemented with a novel feature fusion method, which can effectively purify specific features, capture common features and aggregate common features with specific features. ([reference for paper](website for journal paper)). 


## Installation

Download CogAware:
```
git clone https://github.com/xxxxxxx/CogAware
```

## Requirements

Code for CogAware was prepared using Python 3.6.13. The major requirements are listed below, and the full list can be found in `environment.yml`:
  
- numpy==1.19.3
- pandas==1.1.5
- torch==1.5.0
- torchvision==0.6.0
- scipy==1.5.4
- scikit-learn==0.24.2  
- nni==2.6.1 

## Create environment

We suggest creating a new virtual environment for a clean installation of all the relevant requirements by following commands:

```
conda env create -n CogAware -f environment.yml
```

## Data

We use the Zurich Cognitive Language Processing Corpus (ZuCo) dataset. ZuCo is the sole comprehensive and publicly accessible dataset that employs cognitive signals for various natural language processing (NLP) tasks.
ZuCo is the sole comprehensive and publicly accessible dataset that employs cognitive signals for various natural language processing (NLP) tasks.

## Preprocessing

The data used for training has to be in the `.mat` format. 
For instance, the experiment reported in our paper uses the ZuCo data. The input file can be downloaded from ([task1-SR/Matlab files/~.mat](https://osf.io/q3zws/#!))
To run the preprocessing, use the following command:
```
python preprocessing.py
```
For simplification, the output data from the preprocessing can be found in the file ``CogAware/word_egg``

ZAB_eeg_word.npy is the concatenated data of eye movement features and EEG features, used for model training and model performance testing, placed in the data folder.

## Run the script

To run the script, use the following command:
```
python train.py
```

## Reproducing the result with Neural Network Intelligence (NNI)

Here, we provide a training method with Neural Network Intelligence (NNI) in local mode. The experiment use local mode under the PyTorch framework.

1. Set the experiment configuration in ``config.yml`` file:
   1. set parameter `searchSpacePath` with the search space setting (the setting used in this paper can be found in ``search_space.json``). 
   2. set parameter `tuner` with the tuning algorithm, in our experiment, `BatchTuner` is chosen.
   3. set parameter `trial` with the execution path of the main program.
   4. set parameter `localConfig` to activate the local GPU setting.
2. Search space configuration (``search_space.json``):
   1. set parameter `_type` with the mode of tuning. For `BatchTuner`, the `_type` is set as `choice`.
   2. in the mode `choice`, models with respect to all hyper-parameter combinations in the `_value` will be tested.
3. Run with hyper-parameters:
   1. run the following command to start the training: 
```
nnictl create --config ~/config.yml
```
Use the nni port to check the training process. Training results of models with all hyper-parameter settings will be saved.

Detailed descriptions of how to use NNI can be found in [NNI Documentation](https://nni.readthedocs.io/zh/stable/index.html)

## License

Our code is released under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Journal

IEEE transactions on affective computing
