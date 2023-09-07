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
from models import STNet
from torchvision.transforms import Compose, RandomApply, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip, Normalize


# Get parser and parse arguments
parser = get_main_parser()
# Add new arguments
parser.add_argument('--model_directory', type=str, default='None', help='Model directory')
args = parser.parse_args()
args_dict = vars(args)

real_exp_name = args.model_directory.split('/')[-1]
args.exp_name = real_exp_name

# Get save path and create it in case it is necessary
save_path = args.model_directory

# Set manual seeds and get cuda
seed_everything(17)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Get dataset from the values defined in args
dataset = get_dataset_from_args(args=args)

# Get dataloaders
train_loader, val_loader, test_loader = dataset.get_pretrain_dataloaders(
    layer = args.prediction_layer,
    batch_size = args.batch_size,
    shuffle = False,
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

# Declare model
model = STNet(args.img_backbone, args.img_use_pretrained, dataset.adata.n_vars).to(device)
model = model.to(device)

# Print number of parameters
print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
criterion = torch.nn.MSELoss()

# Load model from file
model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pt')))
model.eval()

# Get predictions
if test_loader is not None:
    metrics, output = test_simple_and_save_output(model, test_loader, criterion, test_transforms)
    aux_adata = dataset.adata[dataset.adata.obs.split == 'test']
    # order = dataset.adata.obs[dataset.adata.obs.split == 'test'].index
else:
    metrics, output = test_simple_and_save_output(model, val_loader, criterion, test_transforms)
    aux_adata = dataset.adata[dataset.adata.obs.split == 'val']
    # order = dataset.adata.obs[dataset.adata.obs.split == 'val'].index

breakpoint()

# Get the unique slide names in the prediction adata and chose the first one
p_slide = aux_adata.obs.slide_id.unique()[-1]
# Get that slide from the global adata
slide_adata = dataset.adata[dataset.adata.obs['slide_id'] == p_slide].copy()
# Modify the uns dictionary to include only the information of the slide
slide_adata.uns['spatial'] = {p_slide: dataset.adata.uns['spatial'][p_slide]}

# Update the slide adata with the predictions and labels
slide_adata.layers['predictions'] = tensor_2_np(preds)
slide_adata.layers['labels'] = tensor_2_np(labels)

# Compute list of PCC values
centered_gt_mat = labels - labels.mean(dim=0)
centered_pred_mat = preds - preds.mean(dim=0)

# Compute pearson correlation with cosine similarity
pcc = torch.nn.functional.cosine_similarity(centered_gt_mat, centered_pred_mat, dim=0)
pcc_np = tensor_2_np(pcc)

# Get a pandas dataframe with the results
results_pcc = pd.DataFrame(
    data = pcc_np,
    index = dataset.adata.var_names,
    columns = ['PCC']
)

# Get best and worst predictions
best_genes = results_pcc.nlargest(2, 'PCC').index.tolist()
worst_genes = results_pcc.nsmallest(2, 'PCC').index.tolist()

# List of plotting genes
plotting_genes = best_genes + worst_genes
plotting_pearson = results_pcc['PCC'][plotting_genes].tolist()

diameter = int(slide_adata.uns['spatial'][p_slide]['scalefactors']['spot_diameter_fullres'])

# Get global image limit coordinates
x_min, y_min = slide_adata.obsm['spatial'].min(axis=0) - 4*diameter
x_max, y_max = slide_adata.obsm['spatial'].max(axis=0) + 4*diameter

# Make figure
fig, ax = plt.subplots(nrows=2, ncols=4)
fig.set_size_inches(14, 6.5)

# Cycle plotting
for i in range(len(plotting_genes)):
    # Define title color
    tit_color = 'k' if i<2 else 'k'

    # Define the normalization to have the same color range in groundtruth and prediction
    gt_min, gt_max = slide_adata[:, [plotting_genes[i]]].layers['labels'].min(), slide_adata[:, [plotting_genes[i]]].layers['labels'].max() 
    pred_min, pred_max = slide_adata[:, [plotting_genes[i]]].layers['predictions'].min(), slide_adata[:, [plotting_genes[i]]].layers['predictions'].max()
    vmin, vmax = min([gt_min, pred_min]), max([gt_max, pred_max])
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # Plot the groundtruth
    # sq.pl.spatial_scatter(slide_adata, color=[plotting_genes[i]], ax=ax[0,i], layer='labels', norm=norm, cmap='jet', crop_coord=(x_min, y_min, x_max, y_max), frameon=True)
    sq.pl.spatial_scatter(slide_adata, color=[plotting_genes[i]], ax=ax[0,i], layer='labels', norm=norm, cmap='jet', img=None, frameon=True, size=1.5)
    ax[0,i].set_title(f'{plotting_genes[i]}', color=tit_color)

    # Plot the prediction
    # sq.pl.spatial_scatter(slide_adata, color=[plotting_genes[i]], ax=ax[1,i], layer='predictions', norm=norm, cmap='jet', crop_coord=(x_min, y_min, x_max, y_max), frameon=True)
    sq.pl.spatial_scatter(slide_adata, color=[plotting_genes[i]], ax=ax[1,i], layer='predictions', norm=norm, cmap='jet', img=None, frameon=True, size=1.5)
    ax[1,i].set_title(f'PCC-Gene $= {round(plotting_pearson[i],3)}$', color=tit_color)

    ax[0,i].spines[['right', 'top','left','bottom']].set_visible(False)
    ax[1,i].spines[['right', 'top','left','bottom']].set_visible(False)

# Format figure
for axis in ax.flatten():
    axis.set_xlabel('')
    axis.set_ylabel('')

ax[0,0].set_ylabel('Ground-truth', fontsize=15)
ax[1,0].set_ylabel('Prediction', fontsize=15)

# fig.suptitle('Best 2 (left) and Worst 2 (right) Predicted Genes', fontsize=20)
fig.tight_layout()
# Save plot 
fig.savefig(os.path.join(save_path, 'best_worst_predictions.png'), dpi=300)
plt.close(fig)


# Plot histogram of the results pcc dataframe
fig, ax = plt.subplots()
results_pcc.hist(bins=50, color='#CC99FF', ax=ax)
ax.grid(False)
ax.set_title('Pearson Correlation for all Genes')
ax.set_xlabel('PCC-Gene')
ax.set_ylabel('# Genes')
ax.spines[['right', 'top']].set_visible(False)
plt.savefig(os.path.join(save_path, 'pcc_gene_histogram.png'), dpi=300)
plt.close()



