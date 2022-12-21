# Multitask NLP


## Configure environment
Install requirements
```commandline
pip install -r requirements.txt
```

## Download data
Pull data from DVC:
```commandline
dvc pull
```

Some data are in archives, run script which unzips them
```commandline
python multitask_nlp\scripts\unzip_data.py
```

## Experiments
Run experiments
```
python -m multitask_nlp.experiments.multitask
```

Single task experiements are located in `multitask_nlp/experiments/single_task_exp` so for example run them as
```
python -m multitask_nlp.experiments.single_task_exp.boolq
```
