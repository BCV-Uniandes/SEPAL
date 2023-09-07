import os
os.environ['USE_PYGEOS'] = '0' # To supress a warning from geopandas
import copy
import json
import wandb
import torch
import numpy as np
import pandas as pd
from utils import *
from models import ImageEncoder
from torchvision.transforms import Compose, RandomApply, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip, Normalize
from datetime import datetime

# Get parser and parse arguments
parser = get_main_parser()
args = parser.parse_args()
args_dict = vars(args)

# If exp_name is None then generate one with the current time
if args.exp_name == 'None':
    args.exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


# Start wandb configs
wandb.init(
    project='Pretrain Spatial Transcriptomics',
    name=args.exp_name,
    config=args_dict
)

print("pretrain")

# Get save path and create is in case it is necessary
save_path = os.path.join('results', args.exp_name)
os.makedirs(save_path, exist_ok=True)

# Save script arguments in json file
with open(os.path.join(save_path, 'script_params.json'), 'w') as f:
    json.dump(args_dict, f, indent=4)

# Set manual seeds and get cuda
seed_everything(17)
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
use_cuda = torch.cuda.is_available()

# Get dataset from the values defined in args
dataset = get_dataset_from_args(args=args)

# Declare train and test loaders
train_dataloader, val_dataloader, test_dataloader = dataset.get_pretrain_dataloaders(
    layer = args.prediction_layer,
    batch_size = args.batch_size,
    shuffle = args.shuffle,
    use_cuda = use_cuda
    )

# Define transformations for the patches
train_transforms = Compose([Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),   
                            RandomHorizontalFlip(p=0.5),
                            RandomVerticalFlip(p=0.5),
                            RandomApply([RandomRotation((90, 90))], p=0.5)])
if args.average_test:
    test_transforms = Compose([Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), EightSymmetry()])
else:
    test_transforms = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Declare device
device = torch.device("cuda" if use_cuda else "cpu")

# Declare model
model = ImageEncoder(backbone=args.img_backbone, use_pretrained=args.img_use_pretrained, latent_dim=dataset.adata.n_vars).to(device)

# Print the number of parameters of the model
num_params = sum(p.numel() for p in model.parameters()) # if p.requires_grad) for just trainable parameters
print(f'Number of model parameters: {num_params}')

# Define the 3 criterions and optimizer
criterion = torch.nn.MSELoss()
try:
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr, momentum=args.momentum)
except:
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)

# Start metric dataframe as None
metric_df = None

# Define dict to know whether to maximize or minimize each metric
max_min_dict = {'PCC-Gene': 'max', 'PCC-Patch': 'max', 'MSE': 'min', 'MAE': 'min', 'R2-Gene': 'max', 'R2-Patch': 'max', 'Global': 'max'}
# Define best values
best_val_optim_metric = -np.inf if max_min_dict[args.optim_metric] == 'max' else np.inf
best_model_wts = copy.deepcopy(model.state_dict())

# Cycle over epochs
for i in range(args.epochs):

    # Train during one epoch
    train_simple(model, train_dataloader, criterion, optimizer, train_transforms)
    # Test in train and val
    train_metric_dict, train_output_dict = test_simple_and_save_output(model, train_dataloader, criterion, test_transforms)
    val_metric_dict, val_output_dict = test_simple_and_save_output(model, val_dataloader, criterion, test_transforms)

    # Update metrics df
    metric_df = update_save_metric_df(metric_df, i, train_metric_dict, val_metric_dict, os.path.join(save_path, 'progress.csv'))
    
    # Make checkpoints every 50 epochs
    if (i+1)%50 == 0:
        checkpoint_wts = copy.deepcopy(model.state_dict())
        torch.save(checkpoint_wts, os.path.join(save_path, f'model_checkpoint_epoch_{i}.pt'))

    # Determine if we got a new best model (robust to minimization or maximization of any metric)
    got_best_min = (max_min_dict[args.optim_metric] == 'min') and (val_metric_dict[args.optim_metric] < best_val_optim_metric)
    got_best_max = (max_min_dict[args.optim_metric] == 'max') and (val_metric_dict[args.optim_metric] > best_val_optim_metric)
    # If we got a new best model, save it and log the metrics in wandb
    if got_best_min or got_best_max:
        best_val_optim_metric = val_metric_dict[args.optim_metric]
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, os.path.join(save_path, 'best_model.pt'))
        wandb.log({f'best_val_{key}':val for key, val in val_metric_dict.items()})

        
# Load best model and test it
model.load_state_dict(best_model_wts)
if test_dataloader is not None:
    test_metric_dict, test_output_dict = test_simple_and_save_output(model, test_dataloader, criterion, test_transforms)
    wandb.log({f'test_{key}':val for key, val in test_metric_dict.items()})


# Print the best result
print('Best results:')
print(metric_df[metric_df['val_PCC-Gene'] == metric_df['val_PCC-Gene'].max()])

wandb.finish()

