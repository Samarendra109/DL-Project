# CSC413 Final Project (Exploring Example Difficulty Metrics for Data Pruning)

This is the code repo for the final project in CSC413 (Spring 2023). In this repo the implementation for [Variance of Gradients (VOG)](https://openreview.net/forum?id=fpJX0O5bWKJ), [Prediction Depth](https://openreview.net/forum?id=WWRBHhH158K) and [Error L2-Norm (EL2N)](https://arxiv.org/abs/2107.07075) is present. Find the implementation under metrics package. 

- clustering.py code is taken from Josue's gist, which can be found [here](https://gist.github.com/JosueCom/7e89afc7f30761022d7747a501260fe3). 
- models.py code is taken from Bohua Peng's repo, which can be found [here](https://github.com/pengbohua/AngularGap). 

Few modifications have been done to the original code to suit our specific use case.

N.B: 
- Don't change the folder structure or files. The relative positioning of the files and the folders is requires fo running the code. The clustering.py, models.py should be at the same level as the metrics package in directory. The ipynb files should be in the same level as the clustering.py and models.py files.
- Ignore the old_metrics_code folder. It contains rough/poc implementations of the metric which is not efficient. This folder can be deleted without any repercussions.
