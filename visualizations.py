from datasets import SpatialDataset
import argparse
import json
import squidpy as sq
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import anndata as ad
import numpy as np
import os
import string
import tqdm
import tifffile

def get_slide_from_collection(collection: ad.AnnData,  slide: str) -> ad.AnnData:
    """
    This function receives a slide name and returns an adata object of the specified slide based on the collection of slides
    in collection.

    Args: 
        collection (ad.AnnData): AnnData object with all the slides.
        slide (str): Name of the slide to get from the collection. Must be in the column 'slide_id' of the obs dataframe of the collection.

    Returns:
        ad.AnnData: An anndata object with the specified slide.
    """

    # Get the slide from the collection
    slide_adata = collection[collection.obs['slide_id'] == slide].copy()
    # Modify the uns dictionary to include only the information of the slide
    slide_adata.uns['spatial'] = {slide: collection.uns['spatial'][slide]}

    # Return the slide
    return slide_adata

def visualize_pepper_filtering(processed_adata: ad.AnnData, slide:str, gene: str) -> None:
    """
    This function uses a gene and slide identifiers to plot the gene expression in the specified slide before and after
    the pepper filtering.

    Args:
        processed_adata (ad.AnnData): An adata with the filtered patient collection
        gene (str): Name of the gene to plot
        slide (str): Name of the slide to plot
    """
    
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(10, 6)

    # Get the slide from the collection
    slide_adata = get_slide_from_collection(processed_adata, slide)

    # Mark the zero entries as nan in the log1p and d_log1p layers
    slide_adata.layers['log1p'][slide_adata.layers['log1p'] == 0] = np.nan
    slide_adata.layers['d_log1p'][slide_adata.layers['d_log1p'] == 0] = np.nan
    
    min_log1p = np.nanmin(slide_adata.layers['log1p'][:, slide_adata.var_names == gene])
    max_log1p = np.nanmax(slide_adata.layers['log1p'][:, slide_adata.var_names == gene])
    min_d_log1p = np.nanmin(slide_adata.layers['d_log1p'][:, slide_adata.var_names == gene])
    max_d_log1p = np.nanmax(slide_adata.layers['d_log1p'][:, slide_adata.var_names == gene])

    gene_min = min(min_log1p, min_d_log1p)
    gene_max = max(max_log1p, max_d_log1p)

    norm = matplotlib.colors.Normalize(vmin=gene_min, vmax=gene_max)
    cmap = matplotlib.colormaps['jet']
    cmap.set_bad('black')

    # Plot selected genes in the specified slides
    sq.pl.spatial_scatter(slide_adata, layer='log1p', color=[gene], ax=ax[0], cmap=cmap, na_color='black', norm=norm, colorbar=False)
    sq.pl.spatial_scatter(slide_adata, layer='d_log1p', color=[gene], ax=ax[1], cmap=cmap, na_color='black', norm=norm, colorbar=False)
    
    ax[0].set_title(f'{gene}\nBefore Pepper Filtering', fontsize=15)
    ax[1].set_title(f'{gene}\nAfter Pepper Filtering', fontsize=15)
    
    # Format figure
    for axis in ax.flatten():
        axis.set_xlabel('')
        axis.set_ylabel('')
        # Turn off all spines
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)

    # Add letter labels to the subplots
    for n, axis in enumerate(ax):    
        axis.text(-0.1, 1.1, string.ascii_uppercase[n], transform=axis.transAxes, size=20, weight='bold')

    cax = fig.add_axes([0.92, 0.2, 0.02, 0.59])
    cbar = fig.colorbar(mappable=cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='$log_2(TPM+1)$')

    # Add legend in axis 0 with black dots for the nan values
    ax[0].scatter([], [], c='black', label='Zero values', s=4)
    ax[0].legend(loc=(0.6, 0.9), frameon=True)

    # fig.tight_layout()
    # Save plot
    os.makedirs('visualizations', exist_ok=True)
    fig.savefig(os.path.join('visualizations', f'pepper_filtering_{gene}.png'), dpi=400)
    plt.close()

def visualize_local_graph(processed_adata: ad.AnnData, slide:str, idx: int, m_hops: int)-> None:
    # Get the slide from the collection
    slide_adata = get_slide_from_collection(processed_adata, slide)
    
    # Get the complete graph
    sq.gr.spatial_neighbors(slide_adata, n_rings=1, coord_type="grid", n_neighs=6)

    # Get the adjacency matrix
    adj_matrix = slide_adata.obsp['spatial_connectivities']

    # Define power matrix
    power_matrix = adj_matrix.copy()
    # Define the output matrix
    output_matrix = adj_matrix.copy()

    # Iterate through the hops
    for i in range(m_hops-1):
        # Compute the next hop
        power_matrix = power_matrix * adj_matrix
        # Add the next hop to the output matrix
        output_matrix = output_matrix + power_matrix

    # Get the indices of the neighbors
    _, idx_neighbors = output_matrix[idx, :].nonzero()

    # If idx is not in the neighbors, add it
    if idx not in idx_neighbors:
        idx_neighbors = np.append(idx_neighbors, idx)
    
    diameter = int(slide_adata.uns['spatial'][slide]['scalefactors']['spot_diameter_fullres'])

    # Get global image limit coordinates
    x_min, y_min = slide_adata.obsm['spatial'].min(axis=0) - diameter
    x_max, y_max = slide_adata.obsm['spatial'].max(axis=0) + diameter

    # Get the coordinates limits of the neighbors
    x_min_neighbors, y_min_neighbors = slide_adata.obsm['spatial'][idx_neighbors, :].min(axis=0) - int(0.7*diameter)
    x_max_neighbors, y_max_neighbors = slide_adata.obsm['spatial'][idx_neighbors, :].max(axis=0) + int(0.7*diameter)
    
    x_min_center, y_min_center = slide_adata.obsm['spatial'][idx, :] - int(0.5*diameter)
    x_max_center, y_max_center = slide_adata.obsm['spatial'][idx, :] + int(0.5*diameter)

    hires_img_path = slide_adata.uns['spatial'][slide]['metadata']["source_image_path"]
    hires_img = tifffile.imread(hires_img_path)

    fig, ax = plt.subplots(ncols=4, figsize=(20, 5))
    sq.pl.spatial_scatter(slide_adata[idx_neighbors, :], connectivity_key="spatial_connectivities", img=True, na_color="lightgrey",
                          frameon=False, ax=ax[0], crop_coord=(x_min, y_min, x_max, y_max))
    sq.pl.spatial_scatter(slide_adata[idx_neighbors, :], connectivity_key="spatial_connectivities", img=hires_img, na_color="lightgrey",
                          frameon=False, ax=ax[1], crop_coord=(x_min_neighbors, y_min_neighbors, x_max_neighbors, y_max_neighbors), scale_factor=1.,
                          edges_color='k', shape='circle', edges_width=2)
    sq.pl.spatial_scatter(slide_adata[idx, :], img=hires_img, color=None, frameon=False, ax=ax[2], crop_coord=(x_min_center, y_min_center, x_max_center, y_max_center), scale_factor=1.)
    ax[3].imshow(slide_adata[idx, :].layers['d_deltas'].T[49:64, :], cmap='jet', vmin=-3, vmax=1.8)
    ax[3].axis('off')
    plt.tight_layout()
    os.makedirs('visualizations', exist_ok=True)
    fig.savefig(os.path.join('visualizations', f'graph_visualization_index_{idx}.png'), dpi=400)
    plt.close()



if __name__ == '__main__':
    # Define a simple parser and add an argument for the config file
    parser = argparse.ArgumentParser(description='Test code for datasets.')
    parser.add_argument('--config', type=str, default='config_dataset.json', help='Path to the config file.')
    args = parser.parse_args()

    # Load the config file
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Define param dict
    param_dict = {
        'cell_min_counts':   config['cell_min_counts'],
        'cell_max_counts':   config['cell_max_counts'],
        'gene_min_counts':   config['gene_min_counts'],
        'gene_max_counts':   config['gene_max_counts'],
        'min_exp_frac':      config['min_exp_frac'],
        'min_glob_exp_frac': config['min_glob_exp_frac'],
        'top_moran_genes':   config['top_moran_genes'],
        'wildcard_genes':    config['wildcard_genes'],
        'combat_key':        config['combat_key'],
        'random_samples':    config['random_samples'],
        'plotting_slides':   config['plotting_slides'],
        'plotting_genes':    config['plotting_genes']
    }

    dataset = SpatialDataset(
        dataset=config['dataset'], 
        param_dict=param_dict, 
        patch_scale=config['patch_scale'], 
        patch_size=config['patch_size'], 
        force_compute=config['force_compute']
    )

    visualize_local_graph(dataset.adata, 'V1_Breast_Cancer_Block_A_Section_1', 2100, 2)

    visualize_pepper_filtering(dataset.adata, 'V1_Breast_Cancer_Block_A_Section_1', 'ENSG00000096006')

