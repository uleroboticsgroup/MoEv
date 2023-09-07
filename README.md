#############Repository structure############

+ DocuMoEv: Containing the documentation of the tool and how it processes the data

+ MoEv3.0: Tool and source code

+ Utils: Contains some scripts that may be useful in data processing but are not part of the tool


To execute the tool:

`python3 start.py -t <data type of datasets>`

**3 types of data:**

*  cic (CICFlowMeter)
*  netflow
*  npy (npy images)

In configuration file `conf.yaml`, we should set up the directories where they are the datasets that we are going to use depending on type of data.

```
npy:
  data_raw: "./imagenes/npy_total_raw_train.npy"
  data_label: "./imagenes/npy_total_label_train.npy"
flows:
  input_path: "/home/test/Descargas/netflow_sampling_250_5-95_test.csv"
  output_path: ""
```



If we want to save the models that we train, we should set up de directory where they will be saved and activate the option in the configuration file.

```
models_path: "./models/"
```

```
Models:
  Save_Models:
    enabled: True
```


We can choose if we train new models or use the models saved previously.
If we choose use the models saved, we have the option to use all tadatset to test the models.

```
testSavedModels:
  enabled: True
  all_dataset: False
```

After, we can choose the models that we want to use. Example:

```
Decision_Tree_Classifier:
  enabled: True
  name: DecisionTreeClassifier
  import: sklearn.tree
```

In every model exists Gridsearch option that is not enabled by default, because it don´t work fine yet.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/uleroboticsgroup/MoEv/Anomalies?labpath=noveltynootebook.ipynb)

