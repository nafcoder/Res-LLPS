# Res-LLPS

### Data availability
All training and independent test datasets are given in [dataset](dataset) folder. The test set features are provided in [ESMC_6B](ESMC_6B) folder. The training set features can be found from https://drive.google.com/drive/folders/1l7fN2uJOlhMiNNx6-OvC7kVcF-NiVmJf?usp=drive_link.

### Environments
OS: Pop!_OS 22.04 LTS


Python version: Python 3.9.19


Used libraries: 
```
numpy==1.26.4
pandas==2.2.1
xgboost==2.0.3
pickle5==0.0.11
scikit-learn==1.2.2
matplotlib==3.8.2
PyQt5==5.15.10
imblearn==0.0
skops==0.9.0
shap==0.45.1
IPython==8.18.1
```

### Reproduce results
In [training](training) and [testing](testing) folders, training and testing reproducable codes are given.

### Reproduce previous paper metrics
In [prev_paper](prev_paper), scripts are provided for reproducing the results of the previous paper.

### Predict
To predict a query protein, please use the [predict](predict) folder. Python 3.10 is required to run the code, as the ESMC_6B embedding generation module depends specifically on this Python version. Also, you will need to install the esm python package by using the command:
```
pip install esm
```
