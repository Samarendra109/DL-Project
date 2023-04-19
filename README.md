# Exploring Example Difficulty Metrics for Data Pruning - CSC413 Final Project
### Anurag Bajpai and Samarendra Chandan Bindu Dash

This is the code repository for the final project in CSC413 (Spring 2023) by Anurag Bajpai and Samarendra Chandan Bindu Dash. It contains the implementation for [Variance of Gradients (VOG)](https://openreview.net/forum?id=fpJX0O5bWKJ), [Prediction Depth](https://openreview.net/forum?id=WWRBHhH158K) and [Error L2-Norm (EL2N)](https://arxiv.org/abs/2107.07075), as well as code to run experiments to replicate the methodology in [Beyond neural scaling laws: beating power law scaling via data pruning](https://arxiv.org/abs/2206.14486). Please find the implementation of the metrics under the metrics package, and the experimentation code in run_experiments.py. 

## Installation
The required packages can be installed using the requirements.txt file:
```bash
pip install -r requirements.txt
```

## Experimentation
The experiments performed for the project can be replicated by running run_experiments.py with the appropriate command line arguments:
```bash
python run_experiment.py --metric=vog --initial_dataset_size=1.0
python run_experiment.py --metric=el2n --probe_epochs=20 --initial_dataset_size=1.0 
python run_experiment.py --metric=pd --initial_dataset_size=1.0
```
This is the full list of command line arguments that can be used to modify each experiment is:
- metric: the metric to use for data pruning. One of 'vog', 'el2n' or 'pd'.
- dataset: the dataset to use for data pruning. One of 'cifar10' or 'svhn'.
- batch_size: the batch size to use. Default value 128.
- probe_epochs: number of epochs to train the probe model for. The report uses 200 for VoG and prediction depth, and 20 for EL2N.
- epochs: number of epochs to train the models for. Default value 200.
- pd_layers: no. of layers to use for calculating prediction depth. Default value 9.
- pd_alpha: fraction of training data to use for clustering for prediction depth. Default value 1.0.
- checkpoint_interval: checkpoint interval for VoG metric calculation. Default value 5.
- num_models: number of models for EL2N metric. Default value 10.
- initial_dataset_size: fraction of the total dataset to keep (between 0 and 1). Default value 1.0
- rng_seed: seed for torch random number generation. Default value 42.

## Acknowledgements
- clustering.py has been adapted from Josue's gist, which can be found [here](https://gist.github.com/JosueCom/7e89afc7f30761022d7747a501260fe3). 
- models.py has been adapted from Bohua Peng's repo, which can be found [here](https://github.com/pengbohua/AngularGap). 


