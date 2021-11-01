# Machine Learning Project 1 (Group: DAT)

### Members of the group:
Alexia Duriez (aduriez11), Desirée Maulà (desireemaula), Tikhon Parshikov (parshikovt)

### Description of the project
This project was conducted as part of the module CS-433 Machine Learning at EPFL. Our team applied different machine learning methods to build a model that predicts the presence or absence of a Higgs Boson after a proton-proton collision, using characteristics of the collision.

### Description of the repository
The final model can be found in run.py, which creates a set of predictions that can be submitted to the AIcrowd platform.

To get predictions you just need:
1. Add files test.csv and train.csv in the folder data
2. Run run.py file ang get file submission.csv with the predictions for the test dataset 

The implementations.py file contains all the different machine learning methods used to build models. 
These methods in other forms that were used during the research can also be found in individual .py files:
* linear_regression.py
* polynomial_regression.py
* ridge_regression.py
* gradient_descent.py
* stochastic_gradient_descent.py
* logistic_regression.py
* penalized_logistic_regression.py

Files preprocessing.py and cross_validation.py consist of functions used in cleaning data and cross_validation, respectively.
File proj1_helpers.py is used for reading files and creating predictions.

Finally, the notebook final_version.ipynb shows the usage of all these files (the process of preprocessing the data in 4 different ways, then trying to apply different algorithms to these 4 data sets)
