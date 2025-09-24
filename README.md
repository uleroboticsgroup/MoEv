<p align="center">
<pre>
╔════════════════════════════════╗
  __  __       ______     
 |  \/  |     |  ____|    
 | \  / | ___ | |____   __
 | |\/| |/ _ \|  __\ \ / /
 | |  | | (_) | |___\ V / 
 |_|  |_|\___/|______\_/  
╚════════════════════════════════╝
</pre>
</p>




[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/uleroboticsgroup/MoEv/sql-mlp-paper?urlpath=%2Fdoc%2Ftree%2FSQLIA_Example.ipynb)
[![python](https://img.shields.io/badge/Python-3.8-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://github.com/uleroboticsgroup/MoEv/blob/main/LICENSE)

## ✨ Description

**MoEv** is a *wrapper* on top of **scikit-learn** that automates the full experimentation pipeline (data loading, preprocessing, normalization, dimensionality reduction, fitting, evaluation, and model saving) through a simple **YAML** file.  

## 📦 Repository Structure

+ MoEv3.0: Tool and source code

+ Utils: Contains some scripts that may be useful in data processing but are not part of the tool

## 🚀 Quick Start

1) **Python 3.8** and recommended dependencies:
```bash
pip install -r requirements.txt
```

2) **Configure your paths** in `conf.yaml`.

3) **Run the tool** specifying the data type:
```bash
python3 start.py -t <type>
# available types:
#  - cic        (CICFlowMeter)
#  - netflow    (NetFlow)
#  - npy        (NumPy arrays/images)
```

💡 You can also try the example notebook on Binder (badge above).

## ⚙️ Configuration


In the `conf.yaml` file, define the paths for your chosen data type:

```yaml
npy:
  data_raw: "./imagenes/npy_total_raw_train.npy"
  data_label: "./imagenes/npy_total_label_train.npy"

flows:
  input_path: "/home/test/Descargas/netflow_sampling_250_5-95_test.csv"
  output_path: ""   # optional output path for processed results
```



### Model Saving

Enable model saving and set the directory:

```yaml
models_path: "./models/"

Models:
  Save_Models:
    enabled: true
```


### Reusing Pre-Trained Models

You can skip training and evaluate previously saved models:

```yaml
testSavedModels:
  enabled: true
  all_dataset: false   # true = evaluate on the whole dataset
```

### Model Selection

Enable/disable algorithms and their imports (example):

```yaml
Decision_Tree_Classifier:
  enabled: true
  name: DecisionTreeClassifier
  import: sklearn.tree
```

In every model exists Gridsearch option that is not enabled by default, because it don´t work fine yet.


## 📄 License

This project is distributed under the **LGPL-3.0** license.  
See the full text here: https://www.gnu.org/licenses/lgpl-3.0.html

