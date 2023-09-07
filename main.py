import os
os.environ['USE_PYGEOS'] = '0' # To supress a warning from geopandas
import copy
import json
import wandb
import torch
import numpy as np
import pandas as pd
from utils import *
from models import MLP
from datetime import datetime
from metrics import get_metrics
from tqdm import tqdm
from models import st_network


# Get parser and parse arguments
parser = get_main_parser()
args = parser.parse_args()
args_dict = vars(args)

# If exp_name is None then generate one with the current time
if args.exp_name == 'None':
    args.exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Start wandb configs
wandb.init(
    project='Spatial Transcriptomics',
    name=args.exp_name,
    config=args_dict
)

# Get save path and create it in case it is necessary
save_path = os.path.join('results', args.exp_name)
os.makedirs(save_path, exist_ok=True)

# Save script arguments in json file
with open(os.path.join(save_path, 'script_params.json'), 'w') as f:
    json.dump(args_dict, f, indent=4)

# Set manual seeds and get cuda
seed_everything(17)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Get dataset from the values defined in args
dataset = get_dataset_from_args(args=args)

# Get dataloaders
train_loader, val_loader, test_loader = dataset.get_graph_dataloaders(
    layer = args.prediction_layer,
    n_hops= args.n_hops,
    backbone=args.img_backbone,
    model_path=args.pretrained_ie_path,
    batch_size=args.batch_size,
    shuffle=args.shuffle
)

# Check input shape for model
initial_shape = train_loader.dataset[0].embeddings.shape[1]
if not args.pos_emb_sum: 
    initial_shape *= 2

if args.h_global[0][0] < 0:
    args.h_global[1][0] = initial_shape

else:
    args.h_global[0][0] = initial_shape

# Declare model
model = st_network(act=args.act, graph_operator=args.graph_operator, h_preprocess=args.h_global[0], h_graph=args.h_global[1], h_pred_head=args.h_global[2], pooling=args.pooling, sum_positions=args.pos_emb_sum)
model = model.to(device)
# Print number of parameters
print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

# Loss function and optimizer
criterion = torch.nn.MSELoss()
try:
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr, momentum=args.momentum)
except:
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)

def calculate_metrics(y_true, y_pred, epoch, type, mask):

    metrics = get_metrics(y_true, y_pred, mask)
    log_dict = { f'{type}_{key}': val for key, val in metrics.items() }
    log_dict['Epoch'] = epoch
    wandb.log(log_dict)
    
    return metrics

def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn):

    all_preds = []
    all_labels = []
    all_masks = []
    running_loss = 0.0
    step = 0

    model.train()

    for _, batch in enumerate(train_loader):

        # Use GPU
        batch.to(device)  

        # Reset gradients
        optimizer.zero_grad() 

        # Passing the node features and the connection info
        gnn_pred = model(batch)
        batch_pred = batch.predictions[batch.ptr[:-1]]
        pred = gnn_pred + batch_pred

        # Get labels
        layer = batch.y
        labels = layer[batch.ptr[:-1]]
        
        # Calculating the loss and gradients
        loss = loss_fn(pred, labels)
        loss.backward()  
        optimizer.step()  

        # Update tracking
        running_loss += loss.item()
        step += 1
        
        # Get mask
        mask = batch.mask[batch.ptr[:-1]]

        # Save results to then compute metrics
        all_preds.append(pred)
        all_labels.append(labels)
        all_masks.append(mask)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Handle the case of predicting deltas
    if hasattr(batch, 'used_mean'):
        all_preds = all_preds + batch.used_mean[:batch.y.shape[1]]
        all_labels = all_labels + batch.used_mean[:batch.y.shape[1]]

    metrics = calculate_metrics(all_labels, all_preds, epoch, "train", all_masks.bool())
    return running_loss/step, metrics

def val_one_epoch(epoch, model, val_loader, loss_fn):

    all_preds = []
    all_labels = []
    all_masks = []
    running_loss = 0.0
    step = 0
    
    model.eval()

    for _, batch in enumerate(val_loader):
        
        batch.to(device)
        gnn_pred = model(batch)
        batch_pred = batch.predictions[batch.ptr[:-1]]
        pred = gnn_pred + batch_pred

        # Get labels
        layer = batch.y
        labels = layer[batch.ptr[:-1]]

        # Calculating the loss 
        loss = loss_fn(torch.squeeze(pred), labels)

        # Update tracking
        running_loss += loss.item()
        step += 1
        
        # Get mask
        mask = batch.mask[batch.ptr[:-1]]

        # Compute metrics
        all_preds.append(pred)
        all_labels.append(labels)
        all_masks.append(mask)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Handle the case of predicting deltas
    if hasattr(batch, 'used_mean'):
        all_preds = all_preds + batch.used_mean[:batch.y.shape[1]]
        all_labels = all_labels + batch.used_mean[:batch.y.shape[1]]

    metrics = calculate_metrics(all_labels, all_preds, epoch, "valid", all_masks.bool())
    return running_loss/step, metrics

def test_model(epoch, model, test_loader):

    all_preds = []
    all_labels = []
    all_masks = []
    
    model.eval()

    for _, batch in enumerate(test_loader):
        
        batch.to(device)
        gnn_pred = model(batch)
        batch_pred = batch.predictions[batch.ptr[:-1]]
        pred = gnn_pred + batch_pred

        # Get labels
        layer = batch.y
        labels = layer[batch.ptr[:-1]]
        
        # Get mask
        mask = batch.mask[batch.ptr[:-1]]

        # Compute metrics
        all_preds.append(pred)
        all_labels.append(labels)
        all_masks.append(mask)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Handle the case of predicting deltas
    if hasattr(batch, 'used_mean'):
        all_preds = all_preds + batch.used_mean[:batch.y.shape[1]]
        all_labels = all_labels + batch.used_mean[:batch.y.shape[1]]

    metrics = calculate_metrics(all_labels, all_preds, epoch, "test", all_masks.bool())

    return metrics


def main():

    best_model_wts = copy.deepcopy(model.state_dict())
    best_optim_metric = np.inf
    new_best = False
    if args.optim_metric == "PCC-Gene":
        best_optim_metric = -np.inf
    
    pbar = tqdm(range(args.epochs))
    for epoch in pbar:

        train_loss, train_metrics = train_one_epoch(epoch, model, train_loader, optimizer, criterion)
        val_loss, val_metrics = val_one_epoch(epoch, model, val_loader, criterion)

        if args.optim_metric == "PCC-Gene":
            if val_metrics["PCC-Gene"] > best_optim_metric:
                best_optim_metric = val_metrics["PCC-Gene"]
                new_best = True
        else:
            if val_metrics[args.optim_metric] < best_optim_metric:
                best_optim_metric = val_metrics[args.optim_metric]  
                new_best = True

        if new_best:
            best_dict = {f'best_val_{key}':val for key, val in val_metrics.items()}
            best_dict['Epoch'] = epoch
            wandb.log(best_dict)
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
            new_best = False

        # Set tqdm description
        pbar.set_description(f'Epoch: {epoch} |  Loss Train: {train_loss:.4f} |  Loss Val: {val_loss:.4f}')
    
    # Load best model and test it
    model.load_state_dict(best_model_wts)
    if test_loader is not None:
        test_metric_dict = test_model(best_dict['Epoch'], model, test_loader)
        wandb.log({f'test_{key}':val for key, val in test_metric_dict.items()})

main()
wandb.finish()

        
    

