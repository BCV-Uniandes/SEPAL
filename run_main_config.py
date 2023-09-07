import json
import subprocess
import argparse
import os

# Get parsed the path of the config file
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_config',     type=str, default='config_dataset.json',    help='Path to the .json file with the configs of the dataset.')
parser.add_argument('--model_config',       type=str, default='config_model.json',      help='Path to the .json file with the configs of the model.')
parser.add_argument('--train_config',       type=str, default='config_train.json',      help='Path to the .json file with the configs of the training.')
parser.add_argument('--model_directory',    type=str, default='None',                   help='Path to the directory of a model.')
args = parser.parse_args()

# Read the dataset, model and train configs
with open(args.dataset_config, 'rb') as f:
    config_params = json.load(f)

with open(args.model_config, 'rb') as f:
    config_params.update(json.load(f))

with open(args.train_config, 'rb') as f:
    config_params.update(json.load(f))

# Set cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = config_params['cuda']
# Used to test an make plot for an arbitrary saved model
if args.model_directory != 'None':
    params_path = os.path.join(args.model_directory, 'script_params.json')
    # Read the params in a dict
    with open(params_path, 'rb') as f:
        params = json.load(f)
    # Update the config_params dict
    config_params.update(params)
    # Add the model directory to the config_params dict
    config_params['model_directory'] = args.model_directory
    config_params['h_global'] = f'//{config_params["h_global"][0]}//{config_params["h_global"][1]}//{config_params["h_global"][2]}'
    config_params['h_global'] = config_params['h_global'].replace(' ','')
    config_params['h_global'] = config_params['h_global'].replace('[','')
    config_params['h_global'] = config_params['h_global'].replace(']','')
    command_list = ['python', 'test_code.py']

# Create the command to run. If sota key is "None" call main.py else call main_sota.py
else:   
    if config_params['sota']=='None':
        command_list = ['python', 'main.py']
    elif config_params['sota']=='nn_baselines':
        command_list = ['python', 'nn_baselines.py']
    elif config_params['sota']=='pretrain':
        command_list = ['python', 'pretrain_backbone.py']
    else:
        command_list = ['python', 'main_STNet.py']

for key, val in config_params.items():
    command_list.append(f'--{key}')
    command_list.append(f'{val}')

# Call subprocess
subprocess.call(command_list)