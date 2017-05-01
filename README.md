# TaQToe

## Getting up and running

_Requires Python 3.4 or higher. You may also want to consider installing this
 inside of a [Conda environment](https://conda.io/docs/py2or3.html#create-a-python-3-5-environment) or [Virtualenv](http://python-guide-pt-br.readthedocs.io/en/latest/dev/virtualenvs/)._

### 1. Clone the repository

```bash
git clone https://github.com/samjabrahams/taqtoe.git

cd taqtoe
```

### 2. Install required dependencies

First make sure that your version of pip is up-to-date:

```
pip install --upgrade pip
```

Then install the requirements listed in `requirements.txt`: 

```
pip install -r requirements.txt
```

### 3. Play some games!

```bash
python main.py
```

## Main usages:

### Play tic-tac-toe with included pre-trained DQN model.

```bash
python main.py
```

### Train you own model from scratch (may take a while to run to completion).

```bash
python main.py -t
# OR
python main.py --train
```

### Play against your custom trained model

```bash
python main.py -c
# OR
python main.py --use_custom_weights
```
