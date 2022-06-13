import json
from pathlib import Path
import os
import sys
path_config_path = Path(os.getcwd()).parent.parent / 'path.json'
with open(path_config_path, 'r') as f:
    path_dict = json.load(f)
config_path = Path(path_dict['config_path'])

# filename
filename = 'config_high_freq_aax'
# change hyperparameter list here
hyperparams_dict = {
    'gamma': [.00001, .0001, .001, .01, .05, .1],
    'base_qty': [.02, .05, .1, ], 
}

with open(config_path / f'{filename}.json', 'r') as f:
    all_config = json.load(f)
market_config = all_config['market_config']


from itertools import product
def my_product(inp):
    return tuple(tuple(zip(inp.keys(), values)) for values in product(*inp.values()))

hyperparams_expand = my_product(hyperparams_dict)

original_log_filename = all_config['LOG']['name']

for i in range(len(hyperparams_expand)):
    hyperparams_set = hyperparams_expand[i]
    for i_hyperparam in range(len(hyperparams_set)):
        hyperparam_name = hyperparams_set[i_hyperparam][0]
        hyperparams_value = hyperparams_set[i_hyperparam][1]
        for i in range(len(market_config)):
            market_config[i][hyperparam_name] = hyperparams_value
    all_config['market_config'] = market_config
    postfix = "_".join([f"{hyperparam_pair[0]}={str(hyperparam_pair[1])}" for hyperparam_pair in hyperparams_set])
    all_config['LOG']['name'] = original_log_filename.split('.')[0] + '_' + postfix + '.log'
    with open(config_path / f'{filename}_{postfix}.json', 'w') as f:
        json.dump(all_config, f, indent=4, sort_keys=True)
