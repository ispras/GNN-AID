# Framework GNN-AID: Graph Neural Network Analysis, Interpretation and Defense

![Image alt](https://github.com/ispras/GNN-AID/blob/develop/GNN-AID%20logo.png)

## Installation
  
GNN-AID was developed and tested under Ubuntu 20.04 and 22, so they suit best for it.  
For another OS consider docker.  

### System requirements

Minimum:

- OS: Linux
- Disk space: 10 GB
- RAM (for framework): 1 GB

Recommended:

- OS: Ubuntu 20.04/22.04
- RAM: 8 GB or more
- GPU: optional. For large-scale experiments and backend only

### Run

You need python of version `3.11` or higher. We advice to create virtual environment with `pip`.
```
python -m pip install --upgrade pip
```
 First, get auxiliary libraries  
```
sudo apt-get install -y build-essential python3-dev libfreetype6-dev pkg-config  
```
Then install all project dependencies
```
pip install -r requirements1.txt
pip install -r requirements2.txt
pip install -r requirements3.txt
```
The 3rd pack of requirements will take around 20 minutes.

### Problems

If you see 139 or 134 error code when run the script, it is likely a compatibility issue. Try the following:

1.  Update video card drivers (if you have decided to use cuda).
2.  Update gcc to the most recent version.
3.  Remove all torch modules that use С++ code.
4.  Install all torch packages again.

## Run in frontend

Suppose you are at the project root folder. Activate virtual environment

```text
source venv/bin/activate
```

Add the current directory to python dependencies

```text
export PYTHONPATH=.
```

Run `main.py` script from `web_interface` package

```text
python web_interface/main.py
```

You will see something like this

```text
======== Running on http://0.0.0.0:5000 ========
(Press CTRL+C to quit)
```

Then go to [127.0.0.1:5000](http://127.0.0.1:5000) in your browser. You should see web-interface is loaded.

## Next steps

We suggest a series of tutorials to learn how to use GNN-AID:

1. Basic GNN training [link](./tutorials/00_basics)
1. Evasion attacks and defense [link](./tutorials/01_evasion_attack_defense)
1. Poison attacks and defense [link](./tutorials/02_poisoning_attack_defense)
1. Privacy attacks and defense [link](./tutorials/03_privacy_attack)
1. Interpreting a GNN Model [link](./tutorials/04_interpretability)
1. A more complex scenario [link](./tutorials/05_complex_scenarios)

You can also check out a short YouTube [video](https://youtu.be/uHxaxLSQ9JM) with demonstration how to operate via GUI.


## Directory structure
```
├── data - storage for datasets raw data
│   ├── example - out-of-the-box example datasets
├── data_info - storages index, updated automatically
├── datasets - storage for preprocessed datasets
├── docs - documentation
├── experiments - section containing experimental scripts
├── explanations - storage for interpretation results
├── gnn_aid - core library
│   ├── attacks - attack methods
│   ├── aux - auxiliary module
│   ├── datasets - dataset handling
│   ├── data_structures - data structures used in the project
│   ├── defenses - defense methods
│   ├── explainers - interpretation methods
│   ├── models_builder - model handling
├── GNN-AID logo.png - logo
├── metainfo - descriptions of function, convolution, and method parameters
├── models - storage for trained models
├── PGE_gen_models - storage for model artifacts used by PGEExplainer
├── pyproject.toml - configuration file for Read the Docs
├── README.md - project description file
├── requirements1.txt - main dependencies
├── requirements2.txt - dependencies required for the documentation
├── requirements3.txt - additional dependencies
├── tests - unit tests
├── tutorials - tutorial examples
├── user_datasets - user-defined datasets
├── user_models_managers - user-defined model managers
├── user_models_obj - user-defined models
├── VERSION - version file
└── web_interface - web interface implementation
    ├── back_front - frontend-backend interaction
    ├── main.py - entry point
    ├── static - frontend code
    │   ├── css - stylesheet files
    │   ├── icons - image assets used in the interface
    │   └── js - JavaScript code
    ├── templates - HTML templates
```

