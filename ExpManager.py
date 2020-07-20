import MDP
import os
import csv

if not os.path.exists("./experiments"):
    os.makedirs("./experiments")

with open('experiments_config.csv') as f:
    reader = csv.DictReader(f)
    for hyper_parameters in reader:
        list_values = [v for v in hyper_parameters.values()]
        name = ','
        name = (name.join(list_values))
        path = f"./experiments/{name}"
        if not os.path.exists(path):
            os.makedirs(path)
        MDP.train_policy(hyper_parameters, path)
