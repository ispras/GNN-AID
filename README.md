# Framework GNN-AID: Graph Neural Network Analysis, Interpretation and Defense

![Image alt](https://github.com/ispras/GNN-AID/blob/develop/GNN-AID%20logo.png)

## Installation
  
GNN-AID was developed and tested under Ubuntu 20.04 and 22, so they suit best for it.  
For another OS consider docker.  

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

Go to `gnn_aid` folder and add it to python dependencies

```text
cd gnn_aid
export PYTHONPATH=.
```

Run `main.py` script

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



