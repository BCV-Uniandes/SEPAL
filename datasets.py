import glob
import scanpy as sc
import anndata as ad
import scipy
import os
os.environ['USE_PYGEOS'] = '0' # To supress a warning from geopandas
import squidpy as sq
import pandas as pd
from tqdm import tqdm
import numpy as np
from anndata.experimental.pytorch import AnnLoader
from sklearn.preprocessing import StandardScaler
from PIL import Image
import matplotlib
import warnings
import gzip
import shutil
import tifffile
import wget
import subprocess
from combat.pycombat import pycombat
import torchvision.models as tmodels
import models
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
import seaborn as sns
from time import time
from datetime import datetime
import json
from torchvision.transforms import Normalize
from typing import Tuple
from torch_geometric.data import Data as geo_Data
from torch_geometric.loader import DataLoader as geo_DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix
import zipfile
import cv2
from torchvision import transforms
import torch
from positional_encodings.torch_encodings import PositionalEncoding2D
import argparse

# Remove the max limit of pixels in a figure
Image.MAX_IMAGE_PIXELS = None
# Set warnings to ignore
warnings.filterwarnings("ignore", message="No data for colormapping provided via 'c'. Parameters 'cmap' will be ignored")
warnings.filterwarnings("ignore", message="Variable names are not unique. To make them unique, call `.var_names_make_unique`.")
warnings.filterwarnings("ignore", message="The 'nopython' keyword argument was not supplied to the 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.")
warnings.filterwarnings("ignore", message="Changing the sparsity structure of a csr_matrix is expensive. lil_matrix is more efficient.")


class STNetReader():
    def __init__(self,
        dataset: str = 'stnet_dataset',
        param_dict: dict = {
            'cell_min_counts':   500,
            'cell_max_counts':   100000,
            'gene_min_counts':   1e3,
            'gene_max_counts':   1e6,
            'min_exp_frac':      0.2,
            'min_glob_exp_frac': 0.6,
            'top_moran_genes':   256,
            'wildcard_genes':    'None',
            'combat_key':        'patient',       
            'random_samples':    -1,              
            'plotting_slides':   'None',          
            'plotting_genes':    'None',          
            }, 
        patch_scale: float = 1.0,
        patch_size: int = 224,
        force_compute: bool = False
        ):
        """
        This is a reader class that can download data, get adata objects and compute a collection of slides into an adata object. It is limited to
        reading and will not perform any processing or filtering on the data. In this particular case, it will read the data from the STNet dataset.

        Args:
            dataset (str, optional): An string encoding the dataset type. In this case only 'stnet_dataset' will work. Defaults to 'stnet_dataset'.
            param_dict (dict, optional): Dictionary that contains filtering and processing parameters. Not used but here for compatibility.
                                        Detailed information about each key can be found in the parser definition over utils.py. 
                                        Defaults to {
                                                'cell_min_counts':   500,
                                                'cell_max_counts':   100000,
                                                'gene_min_counts':   1e3,
                                                'gene_max_counts':   1e6, 
                                                'min_exp_frac':      0.2,
                                                'min_glob_exp_frac': 0.6,
                                                'top_moran_genes':   256,
                                                'wildcard_genes':    'None',
                                                'combat_key':        'patient',
                                                'random_samples':    -1,
                                                'plotting_slides':   'None',
                                                'plotting_genes':    'None',
                                                }.
            patch_scale (float, optional): The scale of the patches to take into account. If bigger than 1, then the patches will be bigger than the original image. Defaults to 1.0.
            patch_size (int, optional): The pixel size of the patches. Defaults to 224.
            force_compute (bool, optional): Whether to force the processing computation or not. Not used but here for compatibility. Defaults to False.
        """

        # We define the variables for the SpatialDataset class
        self.dataset = dataset
        self.param_dict = param_dict
        self.patch_scale = patch_scale
        self.patch_size = patch_size
        self.force_compute = force_compute
        self.hex_geometry = False if self.dataset == 'stnet_dataset' else True

        # We get the dict of split names
        self.split_names = self.get_split_names()
        # We download the data if it is not already downloaded
        self.download_path = self.download_data()
        # Get the dataset path or create one
        self.dataset_path = self.get_or_save_dataset_path()

    def get_split_names(self) -> dict:
        """
        This function uses the self.dataset variable to return a dictionary of names
        if the data split. 
        Returns:
            dict: Dictionary of data names for train, validation and test in lists.
        """
        
        # Get names dictionary
        names_dict = {
            'train': ["BC23287_C1","BC23287_C2","BC23287_D1","BC23450_D2","BC23450_E1",
                      "BC23450_E2","BC23944_D2","BC23944_E1","BC23944_E2","BC24220_D2",
                      "BC24220_E1","BC24220_E2","BC23567_D2","BC23567_E1","BC23567_E2",
                      "BC23810_D2","BC23810_E1","BC23810_E2","BC23903_C1","BC23903_C2",
                      "BC23903_D1","BC24044_D2","BC24044_E1","BC24044_E2","BC24105_C1",
                      "BC24105_C2","BC24105_D1","BC23269_C1","BC23269_C2","BC23269_D1",
                      "BC23272_D2","BC23272_E1","BC23272_E2","BC23277_D2","BC23277_E1",
                      "BC23277_E2","BC23895_C1","BC23895_C2","BC23895_D1","BC23377_C1",
                      "BC23377_C2","BC23377_D1","BC23803_D2","BC23803_E1","BC23803_E2"],
            'val':   ["BC23901_C2","BC23901_D1","BC24223_D2","BC24223_E1","BC24223_E2",
                      "BC23270_D2","BC23270_E1","BC23270_E2","BC23209_C1","BC23209_C2",
                      "BC23209_D1"],
            'test':  ["BC23268_C1","BC23268_C2","BC23268_D1","BC23506_C1","BC23506_C2",
                      "BC23506_D1","BC23508_D2","BC23508_E1","BC23508_E2","BC23288_D2",
                      "BC23288_E1","BC23288_E2"] 
        }

        # Print the names of the datasets
        print(f'Loading {self.dataset} dataset with the following data split:')
        for key, value in names_dict.items():
            print(f'{key} data: {value}')

        return names_dict 

    def download_data(self) -> str:
        """
        This function downloads the data of the original STNet from https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/29ntw7sh4r-5.zip
        using wget to the data/STNet_data directory. Then it unzips the file and deletes the zip file. This function returns a string with the path where the data is stored.

        Returns:
            str: Path to the data directory.
        """
        # Use wget to download the data
        if not os.path.exists(os.path.join('data', 'STNet_data')):
            os.makedirs(os.path.join('data', 'STNet_data'), exist_ok=True)
            wget.download('https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/29ntw7sh4r-5.zip', out=os.path.join('data', 'STNet_data',"29ntw7sh4r-5.zip"))
        
            # Unzip the file in a folder with an understandable name
            os.makedirs(os.path.join('data', 'STNet_data', 'unzipped_STNet_data'))
            with zipfile.ZipFile(os.path.join("data", "STNet_data", "29ntw7sh4r-5.zip"), 'r') as zip_ref:
                zip_ref.extractall(os.path.join('data', 'STNet_data', 'unzipped_STNet_data'))
            
            # Delete the zip file
            os.remove(os.path.join("data", "STNet_data", "29ntw7sh4r-5.zip"))

            # There is an extra folder inside the unzipped folder. We move the files to the unzipped folder.
            files = os.listdir(os.path.join("data", "STNet_data", "unzipped_STNet_data", "Human breast cancer in situ capturing transcriptomics"))
            for file in files:
                shutil.move(os.path.join("data", "STNet_data", "unzipped_STNet_data", "Human breast cancer in situ capturing transcriptomics",file),os.path.join("data", "STNet_data", "unzipped_STNet_data"))

            # We delete the extra folder           
            shutil.rmtree(os.path.join("data", "STNet_data", "unzipped_STNet_data", "Human breast cancer in situ capturing transcriptomics"))

            # Create folders in STNet_data for count_matrix, histology_image, spot_coordinates and tumor_annotation
            folder_names = ['count_matrix', 'histology_image', 'spot_coordinates', 'tumor_annotation']
            [os.makedirs(os.path.join('data', 'STNet_data', f), exist_ok=True) for f in folder_names]

            # move the metadata csv to STNet_data
            shutil.move(os.path.join("data", "STNet_data", "unzipped_STNet_data", "metadata.csv"), os.path.join("data", "STNet_data")) 

            # Read the metadata csv
            metadata = pd.read_csv(os.path.join('data', 'STNet_data', 'metadata.csv'))

            # Iterate over the folder names to unzip the files in the corresponding folder
            for f in folder_names:
                # get filenames from the metadata column
                file_names = metadata[f]
                # If f is histology_image move the files to the histology_image folder
                if f == 'histology_image':
                    [shutil.move(os.path.join("data", "STNet_data", "unzipped_STNet_data", fn),os.path.join("data", "STNet_data", f, fn)) for fn in file_names]
                # If any other folder, extract the .gz files
                else:
                    for fn in file_names:
                        with gzip.open(os.path.join("data", "STNet_data", "unzipped_STNet_data", fn), 'rb') as f_in:
                            with open(os.path.join("data", "STNet_data", "unzipped_STNet_data",fn[:-3]), 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                    # move the files to the corresponding folder
                    [shutil.move(os.path.join("data", "STNet_data", "unzipped_STNet_data", fn[:-3]),os.path.join("data", "STNet_data", f, fn[:-3])) for fn in file_names]

            # We delete the unzipped folder
            shutil.rmtree(os.path.join("data", "STNet_data", "unzipped_STNet_data"))
            
        return os.path.join('data', 'STNet_data') 

    def get_or_save_dataset_path(self) -> str:
        """
        This function saves the parameters of the dataset in a dictionary on a path in the
        processed_dataset folder. The path is returned.

        Returns:
            str: Path to the saved parameters.
        """

        # Get all the class attributes of the current dataset
        curr_dict = self.__dict__.copy()
        
        # Delete some keys from dictionary in order to just leave the class parameters
        curr_dict.pop('force_compute', None)
        curr_dict.pop('plotting_genes', None)
        curr_dict.pop('plotting_slides', None)


        # Define parent folder of all saved datasets
        parent_folder = "processed_" + self.download_path

        # Get the filenames of the parameters of all directories in the parent folder
        filenames = glob.glob(os.path.join(parent_folder, '**', 'parameters.json'), recursive=True)

        # Iterate over all the filenames and check if the parameters are the same
        for filename in filenames:
            with open(filename, 'r') as f:
                # Load the parameters of the dataset
                saved_params = json.load(f)
                # Check if the parameters are the same
                if saved_params == curr_dict:
                    print(f'Parameters already saved in {filename}')
                    return os.path.dirname(filename)

        # If the parameters are not saved, then save them
        # Define directory path to save data
        save_path = os.path.join("processed_" + self.download_path, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        # Make directory if it does not exist
        os.makedirs(save_path, exist_ok=True)

        # Save json
        with open(os.path.join(save_path, 'parameters.json'), 'w') as f:
            json.dump(curr_dict, f, sort_keys=True, indent=4)


        print(f'Parameters not found so this set of parameters is saved in {save_path}')

        return save_path

    def get_adata_for_slide(self, slide_id: str) -> ad.AnnData:
        """
        This function loads the data from the given patient_id and replicate and returns an AnnData object all the with relevant information.
        No image patch information is added by this function. It also computes the quality control metrics of the adata object inplace.
        Finally it uses the compute_moran function to compute and add to the var attribute various statistics related to the Moran's I test.
        In case the data is already computed, it is loaded from the processed_data folder.

        Args:
            patient_id (str): The patient id of the patient to load the data from.
            replicate (str): The replicate to load the data from.
        """
        # Get the patient and replicate from the slide name
        patient_id, replicate = slide_id.split('_')

        # Read the metadata csv
        metadata = pd.read_csv(os.path.join(self.download_path, 'metadata.csv'))

        # Get the row of the patient_id and replicate in the metadata
        slide_row = metadata[(metadata['patient'] == patient_id) & (metadata['replicate'] == replicate)]
        # Get the paths to the files to load
        path_dict = {
            'count_matrix': os.path.join(self.download_path, 'count_matrix', slide_row.count_matrix.item()[:-3]),
            'tumor_annotation': os.path.join(self.download_path, 'tumor_annotation', slide_row.tumor_annotation.item()[:-3]),
            'spot_coordinates': os.path.join(self.download_path, 'spot_coordinates', slide_row.spot_coordinates.item()[:-3]),
            'histology_image': os.path.join(self.download_path, 'histology_image', slide_row.histology_image.item())
        }
        
        # We load the count matrix, tumor annotation, spot coordinates and histology image
        count_matrix = pd.read_csv(path_dict['count_matrix'], index_col = 0, sep='\t', header=0, engine="pyarrow")
        tumor_annotation = pd.read_csv(path_dict['tumor_annotation'], index_col = 0, sep='\t', header=0)
        spot_coordinates = pd.read_csv(path_dict['spot_coordinates'], index_col = 0)
        histology_image = plt.imread(path_dict['histology_image'])

        # Correct tumor_annotation columns by shifting them one to the right
        tumor_annotation.columns = [tumor_annotation.index.name] + tumor_annotation.columns[:-1].to_list()
        tumor_annotation.index.name = None

        # Round the 'xcoord' and 'ycoord' columns of the tumor_annotation to integers
        tumor_annotation['xcoord'] = tumor_annotation['xcoord'].round().astype(int)
        tumor_annotation['ycoord'] = tumor_annotation['ycoord'].round().astype(int)

        # Update tumor_annotation index with the rounded xcoord and ycoord
        tumor_annotation.index = [f'{i.split("_")[0]}_{tumor_annotation.loc[i, "xcoord"]}_{tumor_annotation.loc[i, "ycoord"]}' for i in tumor_annotation.index]

        # Standardize the index of the count_matrix, tumor_annotation and spot_coordinates (format patient_id_replicate_x_y)
        tumor_annotation.index = [f'{patient_id}_{i}' for i in tumor_annotation.index]
        count_matrix.index = [f'{patient_id}_{replicate}_{i.replace("x", "_")}' for i in count_matrix.index]  
        spot_coordinates.index = [f'{patient_id}_{replicate}_{i.replace("x", "_")}' for i in spot_coordinates.index]

        # We compute the intersection between the indexes of the count_matrix, tumor_annotation and spot_coordinates
        intersection_idx = count_matrix.index.intersection(spot_coordinates.index).intersection(tumor_annotation.index)

        # Refine tumor_annotation, count_matrix and spot_coordinates to only contain spots that are in intersection_idx
        tumor_annotation = tumor_annotation.loc[intersection_idx]
        count_matrix = count_matrix.loc[intersection_idx]
        spot_coordinates = spot_coordinates.loc[intersection_idx]
 

        #### Declare obs dataframe
        obs_df = pd.DataFrame({
            'patient': patient_id,
            'replicate': replicate,
            'array_row': tumor_annotation['ycoord'],
            'array_col': tumor_annotation['xcoord'],
            'tumor': tumor_annotation['tumor']=='tumor'
        })
        # Set the index name to spot_id
        obs_df.index.name = 'spot_id'

        #### Get the var_df from the count_matrix
        var_df = count_matrix.columns.to_frame()
        var_df.index.name = 'gene_ids'
        var_df.columns = ['gene_ids']

        #### Declare uns,spatial,sample,metadata dictionary 
        metadata_dict = {
            'chemistry_description': "Spatial Transcriptomics",
            'software_version': 'NA',
            'source_image_path': path_dict['histology_image']
        }

        #### Declare uns,spatial,sample,images dictionary
        # Reshape histology image to lowres (600, 600, 3) and hires (2000, 2000, 3)
        # Read image into PIL
        histology_image = Image.fromarray(histology_image)
        # Resize to lowres
        histology_image_lowres = histology_image.resize((600, int(600*(histology_image.size[1]/histology_image.size[0]))))
        # Resize to hires
        histology_image_hires = histology_image.resize((2000, int(2000*(histology_image.size[1]/histology_image.size[0]))))
        # Convert to numpy array
        histology_image_lowres = np.array(histology_image_lowres)
        histology_image_hires = np.array(histology_image_hires)
        # Create images dictionary
        images_dict = {
            'hires': histology_image_hires,
            'lowres': histology_image_lowres
        }

        # Declare uns,spatial,sample,scalefactors dictionary
        # NOTE: We are trying to compute the scalefactors from the histology image
        scalefactors_dict = {
            'fiducial_diameter_fullres': 'NA',
            'spot_diameter_fullres': 300.0, # This diameter was adjusted by looking at the scatter plot of the spot coordinates
            'tissue_hires_scalef': 2000/histology_image.size[0],
            'tissue_lowres_scalef': 600/histology_image.size[0]
        }

        # Declare uns dictionary
        uns_dict = {
            'spatial': {
                slide_id: {
                    'metadata': metadata_dict,
                    'scalefactors': scalefactors_dict,
                    'images': images_dict
                }

            },
            'cancer_type': slide_row.type.item()
        }

        obsm_dict = {
            'spatial': spot_coordinates.values
        }

        # Declare a scipy sparse matrix from the count matrix
        count_matrix = scipy.sparse.csr_matrix(count_matrix)

        # We create the AnnData object
        adata = ad.AnnData( X = count_matrix,
                            obs = obs_df,
                            var = var_df,
                            obsm = obsm_dict,
                            uns = uns_dict,
                            dtype=np.float32)

        return adata
    
    def get_patches(self, adata: ad.AnnData) -> ad.AnnData:
        """
        This function gets the image patches around a sample center accordingly to a defined scale and then adds them to an observation metadata matrix called 
        adata.obsm[f'patches_scale_{self.patch_scale}'] in the original anndata object. The added matrix has as rows each observation and in each column a pixel of the flattened
        patch.

        Args:
            adata (ad.AnnData): Original anndata object to get the parches. Must have the route to the super high resolution image.
        
        Returns:
            ad.AnnData: An anndata object with the flat patches added to the observation metadata matrix adata.obsm[f'patches_scale_{self.patch_scale}'].
        """
        # Get the name of the sample
        sample_name = list(adata.uns['spatial'].keys())[0]
        # Get the path and read the super high resolution image
        hires_img_path = adata.uns['spatial'][sample_name]['metadata']["source_image_path"]

        # Read the full hires image into numpy array
        hires_img = cv2.imread(hires_img_path)
        # Pass from BGR to RGB
        hires_img = cv2.cvtColor(hires_img, cv2.COLOR_BGR2RGB)
        # Get the spatial coordinates of the centers of the spots
        coord =  pd.DataFrame(adata.obsm['spatial'], columns=['x_coord', 'y_coord'], index=adata.obs_names)

        # Get the size of the window to get the patches
        org_window = int(adata.uns['spatial'][sample_name]['scalefactors']['spot_diameter_fullres']) 
        window = int(org_window * self.patch_scale)
        # If the window is odd, then add one to make it even
        if window % 2 == 1:
            window += 1
        
        # If the window is bigger than the original window, then the image must be padded
        if window > org_window:
            # Get the difference between the original window and the new window
            diff = window - org_window
            # Pad the image
            hires_img = np.pad(hires_img, ((diff//2, diff//2), (diff//2, diff//2), (0, 0)), mode='symmetric')
            # Update the coordinates to the new padded image
            coord['x_coord'] = coord['x_coord'] + diff//2
            coord['y_coord'] = coord['y_coord'] + diff//2

        # Define zeros matrix to store the patches
        flat_patches = np.zeros((adata.n_obs, window**2*3), dtype=np.uint8)

        # Iterate over the coordinates and get the patches
        for i, (x, y) in enumerate(coord.values):
            # Get the patch
            x = int(x)
            y = int(y)
            patch = hires_img[y - (window//2):y + (window//2), x - (window//2):x + (window//2), :]
            # Flatten the patch
            flat_patches[i,:] = patch.flatten()

        # Add the flat crop matrix to a layer in a data
        adata.obsm[f'patches_scale_{self.patch_scale}'] = flat_patches

        return adata

    def get_adata_collection(self) -> ad.AnnData:
        """
        This function reads all the adata objects for the slides in the splits and returns a concatenated AnnData object with all the slides.
        In the adata.obs dataframe the columns 'slide_id' and 'split' are added to identify the slide and the split of each observation.
        Also in the var dataframe the column 'exp_frac' is added with the fraction of cells expressing each gene.
        This 'exp_frac' column is the minimum expression fraction of the gene in all the slides.

        Returns:
            ad.AnnCollection: AnnCollection object with all the slides as AnnData objects.
        """

        # Declare patient adata list
        slide_adata_list = []
        slide_id_list = []

        # Iterate over the slide ids of the splits to get the adata for each slide
        print("The first time running this function will take around 10 minutes to read the data in adata format.")
        for key, value in self.split_names.items():
            print(f'Loading {key} data')
            for slide_id in tqdm(value):
                if not os.path.exists(os.path.join(self.download_path,"adata",f'{slide_id}.h5ad')):
                    # Get the adata for the slide
                    adata = self.get_adata_for_slide(slide_id)
                    # Add the patches to the adata
                    adata = self.get_patches(adata)
                    # Add the slide id to a column in the obs
                    adata.obs['slide_id'] = slide_id
                    # Add the key to a column in the obs
                    adata.obs['split'] = key
                    # Add a unique ID column to the observations to be able to track them when in cuda
                    adata.obs['unique_id'] = adata.obs.index
                    # Change the var_names to the gene_ids
                    adata.var_names = adata.var['gene_ids']
                    # Drop the gene_ids column
                    adata.var.drop(columns=['gene_ids'], inplace=True)
                    # Save adata
                    os.makedirs(os.path.join(self.download_path, "adata"), exist_ok=True)
                    adata.write_h5ad(os.path.join(self.download_path, "adata", f"{slide_id}.h5ad"))
                else:
                    # Load adata
                    adata = ad.read_h5ad(os.path.join(self.download_path, "adata", f"{slide_id}.h5ad"))
                    # This overwrites the split column in the adata to be robust to changes in the splits names
                    adata.obs['split'] = key
                    # If the saved adata does not have the patches, then add them and overwrite the saved adata
                    if f'patches_scale_{self.patch_scale}' not in adata.obsm.keys():
                        # Add the patches to the adata
                        adata = self.get_patches(adata)
                        # Save adata
                        adata.write_h5ad(os.path.join(self.download_path, "adata", f"{slide_id}.h5ad"))
                    # Finally, just leave the patches of the scale that is being used. Remove the rest
                    for obsm_key in list(adata.obsm.keys()):
                        if not(obsm_key in [f'patches_scale_{self.patch_scale}', 'spatial']):
                            adata.obsm.pop(obsm_key) 
                # Add the adata to the list
                slide_adata_list.append(adata)
                # Add the slide id to the list
                slide_id_list.append(slide_id)
        
        # Concatenate all the patients in a single AnnCollection object
        slide_collection = ad.concat(
            slide_adata_list,
            join='inner',
            merge='same'
        )

        # Define a uns dictionary of the collection
        slide_collection.uns = {
            'spatial': {
                slide_id_list[i]: p.uns['spatial'][slide_id_list[i]] for i, p in enumerate(slide_adata_list)
            }
        }

        # Return the patient collection
        return slide_collection

class VisiumReader():
    def __init__(self,
        dataset: str = 'V1_Breast_Cancer_Block_A',
        param_dict: dict = {
            'cell_min_counts':   1000,
            'cell_max_counts':   100000,
            'gene_min_counts':   1e3,
            'gene_max_counts':   1e6,
            'min_exp_frac':      0.8,
            'min_glob_exp_frac': 0.8,
            'top_moran_genes':   256,
            'wildcard_genes':    'None',        
            'combat_key':        'slide_id',    
            'random_samples':    -1,            
            'plotting_slides': 'None',          
            'plotting_genes': 'None',           
                            }, 
        patch_scale: float = 1.0,
        patch_size: int = 224,
        force_compute: bool = False
        ):
        """
        This is a reader class that can download data, get adata objects and compute a collection of slides into an adata object. It is limited to
        reading and will not perform any processing or filtering on the data. In this particular case, it will read the data from Visium datasets.

        Args:
            dataset (str, optional): An string encoding the dataset type. Defaults to 'V1_Breast_Cancer_Block_A'.
            param_dict (dict, optional): Dictionary that contains filtering and processing parameters. Not used but here for compatibility.
                                        Detailed information about each key can be found in the parser definition over utils.py. 
                                        Defaults to {
                                                'cell_min_counts':   1000,
                                                'cell_max_counts':   100000,
                                                'gene_min_counts':   1e3,
                                                'gene_max_counts':   1e6, 
                                                'min_exp_frac':      0.8,
                                                'min_glob_exp_frac': 0.8,
                                                'top_moran_genes':   256,
                                                'wildcard_genes':    'None',
                                                'combat_key':        'slide_id',
                                                'random_samples':    -1,
                                                'plotting_slides':   'None',
                                                'plotting_genes':    'None',
                                                }.
            patch_scale (float, optional): The scale of the patches to take into account. If bigger than 1, then the patches will be bigger than the original image. Defaults to 1.0.
            patch_size (int, optional): The pixel size of the patches. Defaults to 224.
            force_compute (bool, optional): Whether to force the processing computation or not. Not used but here for compatibility. Defaults to False.
        """
        # We define the variables for the SpatialDataset class
        self.dataset = dataset
        self.param_dict = param_dict
        self.patch_scale = patch_scale
        self.patch_size = patch_size
        self.force_compute = force_compute
        self.hex_geometry = False if self.dataset == 'stnet_dataset' else True

        # We get the dict of split names
        self.split_names = self.get_split_names()
        # We download the data if it is not already downloaded
        self.download_path = self.download_data()
        # Get the dataset path or create one
        self.dataset_path = self.get_or_save_dataset_path()

    def get_split_names(self) -> dict:
        """
        This function uses the self.dataset variable to return a dictionary of names
        if the data split. The train dataset consists of the first section of the cut.
        While the validation dataset uses the second section. The names are returned in lists
        for compatibility with the STNet dataset class. For this case the test key has no values
        (empty list).

        Returns:
            dict: Dictionary of data names for train, validation and test in lists.
        """
        # Define train and test data names
        train_data = {
            "V1_Mouse_Brain_Sagittal_Anterior":     "V1_Mouse_Brain_Sagittal_Anterior",
            "V1_Adult_Mouse_Brain_Coronal":         "V1_Adult_Mouse_Brain_Coronal_Section_1",
            "V1_Breast_Cancer_Block_A":             "V1_Breast_Cancer_Block_A_Section_1",
            "V1_Mouse_Brain_Sagittal_Posterior":    "V1_Mouse_Brain_Sagittal_Posterior",
            "V1_Human_Brain":                       "V1_Human_Brain_Section_1"
            }
        
        val_data = {
            "V1_Mouse_Brain_Sagittal_Anterior":     "V1_Mouse_Brain_Sagittal_Anterior_Section_2",
            "V1_Adult_Mouse_Brain_Coronal":         "V1_Adult_Mouse_Brain_Coronal_Section_2",
            "V1_Breast_Cancer_Block_A":             "V1_Breast_Cancer_Block_A_Section_2",
            "V1_Mouse_Brain_Sagittal_Posterior":    "V1_Mouse_Brain_Sagittal_Posterior_Section_2",
            "V1_Human_Brain":                       "V1_Human_Brain_Section_2"
            }
        
        # Get names dictionary
        names_dict = {
            'train': [train_data[self.dataset]],
            'val': [val_data[self.dataset]],
            'test': [] 
        }

        # Print the names of the datasets
        print(f'Loading {self.dataset} dataset with the following data split:')
        for key, value in names_dict.items():
            print(f'{key} data: {value}')

        return names_dict

    def download_data(self) -> str:
        """
        This function downloads the visium data specified and downloads it into
        os.path.join('data', 'Visium_data', self.dataset).
        This function returns a string with the path where the data is stored.

        Returns:
            str: Path to the data directory.
        """
        
        # If the path of the data does not exist, then download the data
        if not os.path.exists(os.path.join('data', 'Visium_data', self.dataset)):
            
            # Iterate over all the names of the split data and download them
            for sample_list in self.split_names.values():
                for element_id in sample_list:
                    _ = sq.datasets.visium(
                        sample_id = element_id,
                        base_dir = os.path.join('data', 'Visium_data', self.dataset),
                        include_hires_tiff = True
                        )
        
        return os.path.join('data', 'Visium_data', self.dataset)

    def get_or_save_dataset_path(self) -> str:
        """
        This function saves the parameters of the dataset in a dictionary on a path in the
        processed_dataset folder. The path is returned.

        Returns:
            str: Path to the saved parameters.
        """

        # Get all the class attributes of the current dataset
        curr_dict = self.__dict__.copy()
        
        # Delete some keys from dictionary in order to just leave the class parameters
        curr_dict.pop('force_compute', None)
        curr_dict.pop('plotting_genes', None)
        curr_dict.pop('plotting_slides', None)


        # Define parent folder of all saved datasets
        parent_folder = "processed_" + self.download_path

        # Get the filenames of the parameters of all directories in the parent folder
        filenames = glob.glob(os.path.join(parent_folder, '**', 'parameters.json'), recursive=True)

        # Iterate over all the filenames and check if the parameters are the same
        for filename in filenames:
            with open(filename, 'r') as f:
                # Load the parameters of the dataset
                saved_params = json.load(f)
                # Check if the parameters are the same
                if saved_params == curr_dict:
                    print(f'Parameters already saved in {filename}')
                    return os.path.dirname(filename)

        # If the parameters are not saved, then save them
        # Define directory path to save data
        save_path = os.path.join("processed_" + self.download_path, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        # Make directory if it does not exist
        os.makedirs(save_path, exist_ok=True)

        # Save json
        with open(os.path.join(save_path, 'parameters.json'), 'w') as f:
            json.dump(curr_dict, f, sort_keys=True, indent=4)


        print(f'Parameters not found so this set of parameters is saved in {save_path}')

        return save_path

    def get_adata_for_slide(self, slide_id: str) -> ad.AnnData:
        """
        This function receives a slide id and returns an adata object by reading the previously downloaded data.

        Args:
            slide_id (str): The id of the slice to read. E.g. V1_Breast_Cancer_Block_A_Section_2

        Returns:
            ad.AnnData: An anndata object with the data of the slice.
        """
        # Get the path for the slice
        path = os.path.join(self.download_path, slide_id)
        # Read the data
        adata = sq.read.visium(path, gex_only=False)
        # Add the path to the original image to the metadata
        adata.uns['spatial'][f'{slide_id}']['metadata']['source_image_path'] = os.path.join(path, 'image.tif')

        return adata
    
    def get_patches(self, adata: ad.AnnData) -> ad.AnnData:
        """
        This function gets the image patches around a sample center accordingly to a defined scale and then adds them to an observation metadata matrix called 
        adata.obsm[f'patches_scale_{self.patch_scale}'] in the original anndata object. The added matrix has as rows each observation and in each column a pixel of the flattened
        patch.

        Args:
            adata (ad.AnnData): Original anndata object to get the parches. Must have the route to the super high resolution image.
        
        Returns:
            ad.AnnData: An anndata object with the flat patches added to the observation metadata matrix adata.obsm[f'patches_scale_{self.patch_scale}'].
        """
        # Get the name of the sample
        sample_name = list(adata.uns['spatial'].keys())[0]
        # Get the path and read the super high resolution image
        hires_img_path = adata.uns['spatial'][sample_name]['metadata']["source_image_path"]

        # Read the full hires image into numpy array
        hires_img = tifffile.imread(hires_img_path)
        # Get the spatial coordinates of the centers of the spots
        coord =  pd.DataFrame(adata.obsm['spatial'], columns=['x_coord', 'y_coord'], index=adata.obs_names)

        # Get the size of the window to get the patches
        org_window = int(adata.uns['spatial'][sample_name]['scalefactors']['spot_diameter_fullres']) 
        window = int(org_window * self.patch_scale)
        # If the window is odd, then add one to make it even
        if window % 2 == 1:
            window += 1
        
        # If the window is bigger than the original window, then the image must be padded
        if window > org_window:
            # Get the difference between the original window and the new window
            diff = window - org_window
            # Pad the image
            hires_img = np.pad(hires_img, ((diff//2, diff//2), (diff//2, diff//2), (0, 0)), mode='symmetric')
            # Update the coordinates to the new padded image
            coord['x_coord'] = coord['x_coord'] + diff//2
            coord['y_coord'] = coord['y_coord'] + diff//2

        # Define zeros matrix to store the patches
        flat_patches = np.zeros((adata.n_obs, window**2*3), dtype=np.uint8)

        # Iterate over the coordinates and get the patches
        for i, (x, y) in enumerate(coord.values):
            # Get the patch
            patch = hires_img[y - (window//2):y + (window//2), x - (window//2):x + (window//2), :]
            # Flatten the patch
            flat_patches[i,:] = patch.flatten()

        # Add the flat crop matrix to a layer in a data
        adata.obsm[f'patches_scale_{self.patch_scale}'] = flat_patches

        return adata    

    def get_adata_collection(self) -> ad.AnnData:
        """
        This function reads all the adata objects for the slides in the splits and returns a concatenated AnnData object with all the slides.
        In the adata.obs dataframe the columns 'slide_id' and 'split' are added to identify the slide and the split of each observation.
        Also in the var dataframe the column 'exp_frac' is added with the fraction of cells expressing each gene.
        This 'exp_frac' column is the minimum expression fraction of the gene in all the slides.

        Returns:
            ad.AnnCollection: AnnCollection object with all the slides as AnnData objects.
        """

        # Declare patient adata list
        slide_adata_list = []
        slide_id_list = []

        # Iterate over the slide ids of the splits to get the adata for each slide
        for key, value in self.split_names.items():
            print(f'Loading {key} data')
            for slide_id in value:
                # Get the adata for the slice
                adata = self.get_adata_for_slide(slide_id)
                # Add the patches to the adata
                adata = self.get_patches(adata)
                # Add the slide id as a prefix to the obs names
                adata.obs_names = [f'{slide_id}_{obs_name}' for obs_name in adata.obs_names]
                # Add the slide id to a column in the obs
                adata.obs['slide_id'] = slide_id
                # Add the key to a column in the obs
                adata.obs['split'] = key
                # Add a unique ID column to the observations to be able to track them when in cuda
                adata.obs['unique_id'] = adata.obs.index
                # Change the var_names to the gene_ids
                adata.var_names = adata.var['gene_ids']
                # Drop the gene_ids column
                adata.var.drop(columns=['gene_ids'], inplace=True)
                # Add the adata to the list
                slide_adata_list.append(adata)
                # Add the slide id to the list
                slide_id_list.append(slide_id)
        
        # Concatenate all the patients in a single AnnCollection object
        slide_collection = ad.concat(
            slide_adata_list,
            join='inner',
            merge='same'
        )

        # Define a uns dictionary of the collection
        slide_collection.uns = {
            'spatial': {
                slide_id_list[i]: p.uns['spatial'][slide_id_list[i]] for i, p in enumerate(slide_adata_list)
            }
        }

        # Return the patient collection
        return slide_collection

# TODO: Think of implementing optional random subsampling of the dataset
class SpatialDataset():
    def __init__(self,
        dataset: str = 'V1_Breast_Cancer_Block_A',
        param_dict: dict = {
            'cell_min_counts':   1000,
            'cell_max_counts':   100000,
            'gene_min_counts':   1e3,
            'gene_max_counts':   1e6,
            'min_exp_frac':      0.8,
            'min_glob_exp_frac': 0.8,
            'top_moran_genes':   256,
            'wildcard_genes':    'None',    
            'combat_key':        'slide_id',
            'random_samples':    -1,        
            'plotting_slides': 'None',      
            'plotting_genes': 'None',       
                            }, 
        patch_scale: float = 1.0,
        patch_size: int = 224,
        force_compute: bool = False
        ):
        """
        This is a spatial data class that contains all the information about the dataset. It will call a reader class depending on the type
        of dataset (by now only visium and STNet are supported). The reader class will download the data and read it into an AnnData collection
        object. Then the dataset class will filter, process and plot quality control graphs for the dataset. The processed dataset will be stored
        for rapid access in the future.

        Args:
            dataset (str, optional): An string encoding the dataset type. Defaults to 'V1_Breast_Cancer_Block_A'.
            param_dict (dict, optional): Dictionary that contains filtering and processing parameters.
                                        Detailed information about each key can be found in the parser definition over utils.py. 
                                        Defaults to {
                                                'cell_min_counts':   1000,
                                                'cell_max_counts':   100000,
                                                'gene_min_counts':   1e3,
                                                'gene_max_counts':   1e6, 
                                                'min_exp_frac':      0.8,
                                                'min_glob_exp_frac': 0.8,
                                                'top_moran_genes':   256,
                                                'wildcard_genes':    'None',
                                                'combat_key':        'slide_id',
                                                'random_samples':    -1,
                                                'plotting_slides':   'None',
                                                'plotting_genes':    'None',
                                                }.
            patch_scale (float, optional): The scale of the patches to take into account. If bigger than 1, then the patches will be bigger than the original image. Defaults to 1.0.
            patch_size (int, optional): The pixel size of the patches. Defaults to 224.
            force_compute (bool, optional): Whether to force the processing computation or not. Defaults to False.
        """

        # We define the variables for the SpatialDataset class
        self.dataset = dataset
        self.param_dict = param_dict
        self.patch_scale = patch_scale
        self.patch_size = patch_size
        self.force_compute = force_compute
        self.hex_geometry = False if self.dataset == 'stnet_dataset' else True

        # We initialize the reader class (Both visium or stnet readers can be returned here)
        self.reader_class = self.initialize_reader()
        # We get the dict of split names
        self.split_names = self.reader_class.split_names
        # We get the dataset download path
        self.download_path = self.reader_class.download_path
        # Get the dataset path
        self.dataset_path = self.reader_class.dataset_path
        # We load or compute the processed adata with patches.
        self.adata = self.load_or_compute_adata()

    def initialize_reader(self):
        """
        This function uses the parameters of the class to initialize the appropiate reader class
        (Visium or STNet) and returns the reader class.
        """

        if self.dataset == 'stnet_dataset':
            reader_class = STNetReader(
                dataset=self.dataset,
                param_dict=self.param_dict,
                patch_scale=self.patch_scale,
                patch_size=self.patch_size,
                force_compute=self.force_compute
            )
        else:
            reader_class = VisiumReader(
                dataset=self.dataset,
                param_dict=self.param_dict,
                patch_scale=self.patch_scale,
                patch_size=self.patch_size,
                force_compute=self.force_compute
            )
        
        return reader_class

    def get_slide_from_collection(self, collection: ad.AnnData,  slide: str) -> ad.AnnData:
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

    def filter_dataset(self, adata: ad.AnnData) -> ad.AnnData:
        """
        This function takes a completely unfiltered and unprocessed (in raw counts) slide collection and filters it
        (both samples and genes) according to self.param_dict. A summary list of the steps is the following:

            1. Filter out observations with total_counts outside the range [cell_min_counts, cell_max_counts].
               This filters out low quality observations not suitable for analysis.
            2. Compute the exp_frac for each gene. This means that for each slide in the collection we compute
                the fraction of the observations that express each gene and then took the minimum across all the slides.
            3. Compute the glob_exp_frac for each gene. This is similar to the exp_frac but instead of computing for each
               slide and taking the minimum we compute it for the whole collection. Slides doesn't matter here.
            4. Filter out genes. This depends on the wildcard_genes parameter and the options are the following:
                
                a. 'None':
                    - Filter out genes that are not expressed in at least min_exp_frac of cells in each slide.
                    - FIlter out genes that are not expressed in at least min_glob_exp_frac of cells in the whole collection.
                    - Filter out genes with counts outside the range [gene_min_counts, gene_max_counts]
                
                b. else:
                    - Read .txt file with wildcard_genes and leave only the genes that are in this file

            5. If there are cells with zero counts in all genes then remove them
            6. Compute quality control metrics

        Args:
            adata (ad.AnnData): An unfiltered and unprocessed (in raw counts) slide collection. Has the patches in obsm.

        Returns:
            ad.AnnData: The filtered adata collection. Patches have not been reshaped here.
        """

        ### Define auxiliary functions

        def get_exp_frac(adata: ad.AnnData) -> ad.AnnData:
            """
            This function computes the expression fraction for each gene in the dataset. Internally it gets the
            expression fraction for each slide and then takes the minimum across all the slides.
            """
            # Get the unique slide ids
            slide_ids = adata.obs['slide_id'].unique()

            # Define zeros matrix of shape (n_genes, n_slides)
            exp_frac = np.zeros((adata.n_vars, len(slide_ids)))

            # Iterate over the slide ids
            for i, slide_id in enumerate(slide_ids):
                # Get current slide adata
                slide_adata = adata[adata.obs['slide_id'] == slide_id, :]
                # Get current slide expression fraction
                curr_exp_frac = np.squeeze(np.asarray((slide_adata.X > 0).sum(axis=0) / slide_adata.n_obs))
                # Add current slide expression fraction to the matrix
                exp_frac[:, i] = curr_exp_frac
            
            # Compute the minimum expression fraction for each gene across all the slides
            min_exp_frac = np.min(exp_frac, axis=1)

            # Add the minimum expression fraction to the var dataframe of the slide collection
            adata.var['exp_frac'] = min_exp_frac

            # Return the adata
            return adata

        def get_glob_exp_frac(adata: ad.AnnData) -> ad.AnnData:
            """
            This function computes the global expression fraction for each gene in the dataset.

            Args:
                adata (ad.AnnData): An unfiltered and unprocessed (in raw counts) slide collection.

            Returns:
                ad.AnnData: The same slide collection with the glob_exp_frac added to the var dataframe.
            """
            # Get global expression fraction
            glob_exp_frac = np.squeeze(np.asarray((adata.X > 0).sum(axis=0) / adata.n_obs))

            # Add the global expression fraction to the var dataframe of the slide collection
            adata.var['glob_exp_frac'] = glob_exp_frac

            # Return the adata
            return adata


        # Start tracking time
        print('Starting data filtering...')
        start = time()

        # Get initial gene and observation numbers
        n_genes_init = adata.n_vars
        n_obs_init = adata.n_obs

        ### Filter out samples:

        # Find indexes of cells with total_counts outside the range [cell_min_counts, cell_max_counts]
        sample_counts = np.squeeze(np.asarray(adata.X.sum(axis=1)))
        bool_valid_samples = (sample_counts > self.param_dict['cell_min_counts']) & (sample_counts < self.param_dict['cell_max_counts'])
        valid_samples = adata.obs_names[bool_valid_samples]

        # Subset the adata to keep only the valid samples
        adata = adata[valid_samples, :].copy()

        ### Filter out genes:

        # Compute the min expression fraction for each gene across all the slides
        adata = get_exp_frac(adata)
        # Compute the global expression fraction for each gene
        adata = get_glob_exp_frac(adata)
        
        # If no wildcard genes are specified then filter genes based in min_exp_frac and total counts
        if self.param_dict['wildcard_genes'] == 'None':
            
            gene_counts = np.squeeze(np.asarray(adata.X.sum(axis=0)))
            
            # Find indexes of genes with total_counts inside the range [gene_min_counts, gene_max_counts]
            bool_valid_gene_counts = (gene_counts > self.param_dict['gene_min_counts']) & (gene_counts < self.param_dict['gene_max_counts'])
            # Find indexes of genes with exp_frac larger than min_exp_frac
            bool_valid_gene_exp_frac = adata.var['exp_frac'] > self.param_dict['min_exp_frac']
            # Find indexes of genes with glob_exp_frac larger than min_glob_exp_frac
            bool_valid_gene_glob_exp_frac = adata.var['glob_exp_frac'] > self.param_dict['min_glob_exp_frac']
            # Find indexes of genes that pass all the filters
            bool_valid_genes = bool_valid_gene_counts & bool_valid_gene_exp_frac & bool_valid_gene_glob_exp_frac
            # Get the valid genes
            valid_genes = adata.var_names[bool_valid_genes]

            # Subset the adata to keep only the valid genes
            adata = adata[:, valid_genes].copy()
        
        # If there are wildcard genes then read them and subset the dataset to just use them
        else:
            # Read valid wildcard genes
            genes = pd.read_csv(self.param_dict['wildcard_genes'], sep=" ", header=None, index_col=False)
            # Turn wildcard genes to pandas Index object
            valid_genes = pd.Index(genes.iloc[:, 0], name='')
            # Subset processed adata with wildcard genes
            adata = adata[:, valid_genes].copy()
        
        ### Remove cells with zero counts in all genes:

        # If there are cells with zero counts in all genes then remove them
        null_cells = adata.X.sum(axis=1) == 0
        if null_cells.sum() > 0:
            adata = adata[~null_cells].copy()
            print(f"Removed {null_cells.sum()} cells with zero counts in all selected genes")
        
        ### Compute quality control metrics:

        # As we have removed the majority of the genes, we recompute the quality metrics
        sc.pp.calculate_qc_metrics(adata, inplace=True, log1p=False, percent_top=None)

        # Print the number of genes and cells that survived the filtering
        print(f'Data filtering took {time() - start:.2f} seconds')
        print(f"Number of genes that passed the filtering:        {adata.n_vars} out of {n_genes_init} ({100*adata.n_vars/n_genes_init:.2f}%)")
        print(f"Number of observations that passed the filtering: {adata.n_obs} out of {n_obs_init} ({100*adata.n_obs/n_obs_init:.2f}%)")

        return adata

    # TODO: Test the complete processing pipeline against R implementation (GeoTcgaData package for TPM)
    # TODO: Update the docstring of this function
    def process_dataset(self, adata: ad.AnnData) -> ad.AnnData:
        """
        This function performs the complete processing pipeline for a dataset. It only computes over the expression values of the dataset
        (adata.X). The processing pipeline is the following:
 
            1. Normalize the data with tpm normalization (tpm layer)
            2. Transform the data with log1p (log1p layer)
            3. Denoise the data with the adaptive median filter (d_log1p layer)
            4. Compute moran I for each gene in each slide and average moranI across slides (add results to .var['d_log1p_moran'])
            5. Filter dataset to keep the top self.param_dict['top_moran_genes'] genes with highest moran I.
            6. Perform ComBat batch correction if specified by the 'combat_key' parameter (c_d_log1p layer)
            7. Compute the deltas from the mean for each gene (computed from log1p layer and c_d_log1p layer if batch correction was performed)
            8. Add a binary mask layer specifying valid observations for metric computation.


        Args:
            adata (ad.AnnData): The AnnData object to process. Must be already filtered.

        Returns:
            ad.Anndata: The processed AnnData object with all the layers and results added.
        """

        ### Define processing functions:

        def tpm_normalization(adata: ad.AnnData, from_layer: str, to_layer: str) -> ad.AnnData:
            """
            This function apply tpm normalization to an AnnData object. It also removes genes that are not fount in the gtf annotation file.
            The counts of the anndata are taken from the layer 'from_layer' and the results are stored in the layer 'to_layer'.
            Args:
                adata (ad.Anndata): The Anndata object to normalize.
                from_layer (str): The layer to take the counts from.
                to_layer (str): The layer to store the results of the normalization.
            Returns:
                ad.Anndata: The normalized Anndata object with TPM values in the .layers[to_layer] attribute.
            """
            
            # Get the number of genes before filtering
            initial_genes = adata.shape[1]

            # Automatically download the gtf annotation file if it is not already downloaded
            if not os.path.exists(os.path.join('data', 'annotations', 'gencode.v43.basic.annotation.gtf.gz')):
                print('Automatically downloading gtf annotation file...')
                os.makedirs(os.path.join('data', 'annotations'), exist_ok=True)
                wget.download(
                    'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/gencode.v43.basic.annotation.gtf.gz',
                    out = os.path.join('data', 'annotations', 'gencode.v43.basic.annotation.gtf.gz'))

            # Define gtf path
            gtf_path = os.path.join('data', 'annotations', 'gencode.v43.basic.annotation.gtf')

            # Unzip the data in annotations folder if it is not already unzipped
            if not os.path.exists(gtf_path):
                with gzip.open(os.path.join('data', 'annotations', 'gencode.v43.basic.annotation.gtf.gz'), 'rb') as f_in:
                    with open(gtf_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

            # Obtain a txt with gene lengths
            gene_length_path = os.path.join('data', 'annotations', 'gene_length.txt')
            if not os.path.exists(gene_length_path):
                command = f'python gtftools.py -l {gene_length_path} {gtf_path}'
                command_list = command.split(' ')
                subprocess.call(command_list)   

            # Upload the gene lengths
            glength_df = pd.read_csv(gene_length_path, delimiter='\t', usecols=['gene', 'merged'])

            # For the gene column, remove the version number
            glength_df['gene'] = glength_df['gene'].str.split('.').str[0]

            # Drop gene duplicates. NOTE: This only eliminates 40/60k genes so it is not a big deal
            glength_df = glength_df.drop_duplicates(subset=['gene'])

            # Find the genes that are in the gtf annotation file
            common_genes=list(set(adata.var_names)&set(glength_df["gene"]))

            # Subset both adata and glength_df to keep only the common genes
            adata = adata[:, common_genes].copy()
            glength_df = glength_df[glength_df["gene"].isin(common_genes)].copy()

            # Reindex the glength_df to genes
            glength_df = glength_df.set_index('gene')
            # Reindex glength_df to adata.var_names
            glength_df = glength_df.reindex(adata.var_names)
            # Assert indexes of adata.var and glength_df are the same
            assert (adata.var.index == glength_df.index).all()

            # Add gene lengths to adata.var
            adata.var['gene_length'] = glength_df['merged'].values

            # Divide each column of the counts matrix by the gene length. Save the result in layer "to_layer"
            adata.layers[to_layer] = adata.layers[from_layer] / adata.var['gene_length'].values.reshape(1, -1)
            # Make that each row sums to 1e6
            adata.layers[to_layer] = adata.layers[to_layer] / (np.sum(adata.layers[to_layer], axis=1).reshape(-1, 1)/1e6)
            # Pass layer to np.array
            adata.layers[to_layer] = np.array(adata.layers[to_layer])

            # Print the number of genes that were not found in the gtf annotation file
            failed_genes = initial_genes - adata.n_vars
            print(f'Number of genes not found in GTF file by TPM normalization: {initial_genes - adata.n_vars} out of {initial_genes} ({100*failed_genes/initial_genes:.2f}%) ({adata.n_vars} remaining)')

            # Return the transformed AnnData object
            return adata

        def log1p_transformation(adata: ad.AnnData, from_layer: str, to_layer: str) -> ad.AnnData:
            """
            Simple wrapper around sc.pp.log1p to transform data from adata.layers[from_layer] with log1p (base 2)
            and save it in adata.layers[to_layer].

            Args:
                adata (ad.AnnData): The AnnData object to transform.
                from_layer (str): The layer to take the data from.
                to_layer (str): The layer to store the results of the transformation.

            Returns:
                ad.AnnData: The transformed AnnData object with log1p transformed data in adata.layers[to_layer].
            """

            # Transform the data with log1p
            transformed_adata = sc.pp.log1p(adata, base= 2.0, layer=from_layer, copy=True)

            # Add the log1p transformed data to adata.layers[to_layer]
            adata.layers[to_layer] = transformed_adata.layers[from_layer]

            # Return the transformed AnnData object
            return adata

        def clean_noise(collection: ad.AnnData, from_layer: str, to_layer: str, n_dist_max: int) -> ad.AnnData:
            """
            This wrapper function computes the adaptive median filter for all the slides in the collection and then concatenates the results
            into another collection. Details of the adaptive median filter can be found in the function adaptive_median_filter_peper function.

            Args:
                collection (ad.AnnData): The AnnData collection to process. Contains all the slides.
                from_layer (str): The layer to compute the adaptive median filter from. Where to clean the noise from.
                to_layer (str): The layer to store the results of the adaptive median filter. Where to store the cleaned data.
                n_dist_max (int): The maximum number of concentric circles to take into account to compute the median. See adaptive_median_filter_peper.

            Returns:
                ad.AnnData: The processed AnnData collection with the results of the adaptive median filter stored in the layer 'to_layer'.
            """

            ### Define cleaning function for single slide:

            def adaptive_median_filter_pepper(adata: ad.AnnData, from_layer: str, to_layer: str, n_dist_max: int) -> ad.AnnData:
                """
                This function computes the adaptive median filter for pair (obs, gene) with a zero value (peper noise) in the layer 'from_layer' and
                stores the result in the layer 'to_layer'. The max window size is n_dist_max. This means the number of concentric circles to take into
                account to compute the median.

                Args:
                    adata (ad.AnnData): The AnnData object to process. Importantly it is only from a single slide. Can not be a collection of slides.
                    from_layer (str): The layer to compute the adaptive median filter from.
                    to_layer (str): The layer to store the results of the adaptive median filter.
                    n_dist_max (int): The maximum number of concentric circles to take into account to compute the median. Analog to max window size.

                Returns:
                    ad.AnnData: The AnnData object with the results of the adaptive median filter stored in the layer 'to_layer'.
                """
                
                # Define original expression matrix
                original_exp = adata.layers[from_layer]

                # Compute the indexes of the top s_max closest neighbors for each observation
                cols=torch.tensor(adata.obs['array_col'], dtype=torch.float)
                rows=torch.tensor(adata.obs['array_row'], dtype=torch.float)
                coordinates = torch.stack([rows, cols], dim=1)
                distances = torch.cdist(coordinates, coordinates, p=2)
                
                # Get unique distances without the 0 distance
                unique_distances = torch.unique(distances, sorted=True)[1:n_dist_max+1]    

                medians = np.zeros((adata.n_obs, n_dist_max, adata.n_vars))

                # Iterate over the unique distances
                for i, distance in enumerate(unique_distances):
                    # Get binary mask of distances equal or less than the current distance
                    mask = distances <= distance
                    # Iterate over observations
                    for j in range(mask.shape[0]):
                        # Get the true indexes of the mask in row j
                        true_idx = torch.where(mask[j, :])[0].numpy()
                        # Get the expression matrix of the neighbors
                        neighbor_exp = original_exp[true_idx, :]
                        # Get the median of the expression matrix
                        median = np.median(neighbor_exp, axis=0)

                        # Store the median in the medians matrix
                        medians[j, i, :] = median
                
                # Also robustly compute the median of the non-zero values for each gene
                general_medians = np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]), 0, original_exp)
                general_medians[np.isnan(general_medians)] = 0.0 # Correct for possible nans

                # Define corrected expression matrix
                corrected_exp = np.zeros_like(original_exp)

                ### Now that all the possible medians are computed. We code for each observation:
                
                # Note: i indexes over observations, j indexes over genes
                for i in range(adata.n_obs):
                    for j in range(adata.n_vars):
                        
                        # Get real expression value
                        z_xy = original_exp[i, j]

                        # Only apply adaptive median filter if real expression is zero
                        if z_xy != 0:
                            corrected_exp[i,j] = z_xy
                            continue
                        
                        else:

                            # Definie initial stage and window size
                            current_stage = 'A'
                            k = 0

                            while True:

                                # Stage A:
                                if current_stage == 'A':
                                    
                                    # Get median value
                                    z_med = medians[i, k, j]

                                    # If median is not zero then go to stage B
                                    if z_med != 0:
                                        current_stage = 'B'
                                        continue
                                    # If median is zero, then increase window and repeat stage A
                                    else:
                                        k += 1
                                        if k < n_dist_max:
                                            current_stage = 'A'
                                            continue
                                        # If we have the biggest window size, then return the median
                                        else:
                                            # NOTE: Big modification to the median filter here. Be careful
                                            # FIXME: Think about the general mean instead of the general median here
                                            # corrected_exp[i,j] = z_med
                                            corrected_exp[i,j] = general_medians[j]
                                            break


                                # Stage B:
                                elif current_stage == 'B':
                                    
                                    # Get window median
                                    z_med = medians[i, k, j]

                                    # If real expression is not peper then return it
                                    if z_xy != 0:
                                        corrected_exp[i,j] = z_xy
                                        break
                                    # If real expression is peper, then return the median
                                    else:
                                        corrected_exp[i,j] = z_med
                                        break

                # Add corrected expression to adata
                adata.layers[to_layer] = corrected_exp

                return adata

            # Print message
            print('Applying adaptive median filter to collection...')

            # Get the unique slides
            slides = np.unique(collection.obs['slide_id'])

            # Define the corrected adata list
            corrected_adata_list = []

            # Iterate over the slides
            for slide in tqdm(slides):
                # Get the adata of the slide
                adata = collection[collection.obs['slide_id'] == slide].copy()
                # Apply adaptive median filter
                adata = adaptive_median_filter_pepper(adata, from_layer, to_layer, n_dist_max)
                # Append to the corrected adata list
                corrected_adata_list.append(adata)
            
            # Concatenate the corrected adata list
            corrected_collection = ad.concat(corrected_adata_list, join='inner', merge='same')
            # Restore the uns attribute
            corrected_collection.uns = collection.uns

            return corrected_collection

        def combat_transformation(adata: ad.AnnData, batch_key: str, from_layer: str, to_layer: str) -> ad.AnnData:
            """
            Batch correction using pycombat. The batches are defined by the batch_key column in adata.obs. The input data for
            the batch correction is adata.layers[from_layer] and the output is stored in adata.layers[to_layer].

            Args:
                adata (ad.AnnData): The AnnData object to transform. Must have log1p transformed data in adata.layers[from_layer].
                batch_key (str): The column in adata.obs that defines the batches.
                from_layer (str): The layer to take the data from.
                to_layer (str): The layer to store the results of the transformation.

            Returns:
                ad.AnnData: The transformed AnnData object with batch corrected data in adata.layers[to_layer].
            """
            # Get expression matrix dataframe
            df = adata.to_df(layer = from_layer).T
            batch_list = adata.obs[batch_key].values.tolist()

            # Apply pycombat batch correction
            corrected_df = pycombat(df, batch_list, par_prior=True)

            # Assign batch corrected expression to .layers[to_layer] attribute
            adata.layers[to_layer] = corrected_df.T

            return adata
        
        def get_deltas(adata: ad.AnnData, from_layer: str, to_layer: str) -> ad.AnnData:
            """
            Compute the deviations from the mean expression of each gene in adata.layers[from_layer] and save them
            in adata.layers[to_layer]. Also add the mean expression of each gene to adata.var[f'{from_layer}_avg_exp'].

            Args:
                adata (ad.AnnData): The AnnData object to update. Must have expression values in adata.layers[from_layer].
                from_layer (str): The layer to take the data from.
                to_layer (str): The layer to store the results of the transformation.

            Returns:
                ad.AnnData: The updated AnnData object with the deltas and mean expression.
            """

            # Get the expression matrix of both train and global data
            glob_expression = adata.to_df(layer=from_layer)
            train_expression = adata[adata.obs['split'] == 'train'].to_df(layer=from_layer)

            # Define scaler
            scaler = StandardScaler(with_mean=True, with_std=False)

            # Fit the scaler to the train data
            scaler = scaler.fit(train_expression)
            
            # Get the centered expression matrix of the global data
            centered_expression = scaler.transform(glob_expression)

            # Add the deltas to adata.layers[to_layer]	
            adata.layers[to_layer] = centered_expression

            # Add the mean expression to adata.var[f'{from_layer}_avg_exp']
            adata.var[f'{from_layer}_avg_exp'] = scaler.mean_

            # Return the updated AnnData object
            return adata

        def compute_moran(adata: ad.AnnData, hex_geometry: bool, from_layer: str) -> ad.AnnData:
            """
            This function cycles over each slide in the adata object and computes the Moran's I for each gene.
            After that, it averages the Moran's I for each gene across all slides and saves it in adata.var[f'{from_layer}_moran'].
            The input data for the Moran's I computation is adata.layers[from_layer].

            Args:
                adata (ad.AnnData): The AnnData object to update. Must have expression values in adata.layers[from_layer].
                from_layer (str): The key in adata.layers with the values used to compute Moran's I.
                hex_geometry (bool): Whether the data is hexagonal or not. This is used to compute the spatial neighbors before computing Moran's I.

            Returns:
                ad.AnnData: The updated AnnData object with the average Moran's I for each gene in adata.var[f'{from_layer}_moran'].
            """
            print(f'Computing Moran\'s I for each gene over each slide using data of layer {from_layer}...')

            # Get the unique slide_ids
            slide_ids = adata.obs['slide_id'].unique()

            # Create a dataframe to store the Moran's I for each slide
            moran_df = pd.DataFrame(index = adata.var.index, columns=slide_ids)

            # Cycle over each slide
            for slide in slide_ids:
                # Get the annData for the current slide
                slide_adata = self.get_slide_from_collection(adata, slide)
                # Compute spatial_neighbors
                if hex_geometry:
                    # Hexagonal visium case
                    sq.gr.spatial_neighbors(slide_adata, coord_type='generic', n_neighs=6)
                else:
                    # Grid STNet dataset case
                    sq.gr.spatial_neighbors(slide_adata, coord_type='grid', n_neighs=8)
                # Compute Moran's I
                sq.gr.spatial_autocorr(
                    slide_adata,
                    mode="moran",
                    layer=from_layer,
                    genes=slide_adata.var_names,
                    n_perms=1000,
                    n_jobs=-1,
                    seed=42
                )

                # Get moran I
                moranI = slide_adata.uns['moranI']['I']
                # Reindex moranI to match the order of the genes in the adata object
                moranI = moranI.reindex(adata.var.index)

                # Add the Moran's I to the dataframe
                moran_df[slide] = moranI

            # Compute the average Moran's I for each gene
            adata.var[f'{from_layer}_moran'] = moran_df.mean(axis=1)

            # Return the updated AnnData object
            return adata

        def filter_by_moran(adata: ad.AnnData, n_keep: int, from_layer: str) -> ad.AnnData:
            """
            This function filters the genes in adata.var by the Moran's I. It keeps the n_keep genes with the highest Moran's I.
            The Moran's I values will be selected from adata.var[f'{from_layer}_moran'].

            Args:
                adata (ad.AnnData): The AnnData object to update. Must have adata.var[f'{from_layer}_moran'].
                n_keep (int): The number of genes to keep.
                from_layer (str): Layer for which the Moran's I was computed the key in adata.var is f'{from_layer}_moran'.

            Returns:
                ad.AnnData: The updated AnnData object with the filtered genes.
            """

            # Assert that the number of genes is at least n_keep
            assert adata.n_vars >= n_keep, f'The number of genes in the AnnData object is {adata.n_vars}, which is less than n_keep ({n_keep}).'

            # Sort the genes by Moran's I
            sorted_genes = adata.var.sort_values(by=f'{from_layer}_moran', ascending=False).index

            # Get genes to keep list
            genes_to_keep = list(sorted_genes[:n_keep])

            # Filter the genes andata object
            adata = adata[:, genes_to_keep]

            # Return the updated AnnData object
            return adata


        ### Now compute all the processing steps
        # NOTE: The d prefix stands for denoised
        # NOTE: The c prefix stands for combat

        # Start the timer and print the start message
        start = time()
        print('Starting data processing...')

        # First add raw counts to adata.layers['counts']
        adata.layers['counts'] = adata.X.toarray()

        # Make TPM normalization
        adata = tpm_normalization(adata, from_layer='counts', to_layer='tpm')

        # Transform the data with log1p (base 2.0)
        adata = log1p_transformation(adata, from_layer='tpm', to_layer='log1p')

        # Denoise the data with pepper noise
        adata = clean_noise(adata, from_layer='log1p', to_layer='d_log1p', n_dist_max=7)

        # Compute average moran for each gene in the layer d_log1p 
        adata = compute_moran(adata, hex_geometry=self.hex_geometry, from_layer='d_log1p')

        # Filter genes by Moran's I
        adata = filter_by_moran(adata, n_keep=self.param_dict['top_moran_genes'], from_layer='d_log1p')

        # If combat key is specified, apply batch correction
        if self.param_dict['combat_key'] != 'None':
            adata = combat_transformation(adata, batch_key=self.param_dict['combat_key'], from_layer='log1p', to_layer='c_log1p')
            adata = combat_transformation(adata, batch_key=self.param_dict['combat_key'], from_layer='d_log1p', to_layer='c_d_log1p')

        # Compute deltas and mean expression for all log1p, d_log1p, c_log1p and c_d_log1p
        adata = get_deltas(adata, from_layer='log1p', to_layer='deltas')
        adata = get_deltas(adata, from_layer='d_log1p', to_layer='d_deltas')
        adata = get_deltas(adata, from_layer='c_log1p', to_layer='c_deltas')
        adata = get_deltas(adata, from_layer='c_d_log1p', to_layer='c_d_deltas')

        # Add a binary mask layer specifying valid observations for metric computation
        adata.layers['mask'] = adata.layers['tpm'] != 0
        # Print with the percentage of the dataset that was replaced
        print('Percentage of imputed observations with median filter: {:5.3f}%'.format(100 * (~adata.layers["mask"]).sum() / (adata.n_vars*adata.n_obs)))

        # Print the number of cells and genes in the dataset
        print(f'Processing of the data took {time() - start:.2f} seconds')
        print(f'The processed dataset looks like this:')
        print(adata)
        
        return adata

    def reshape_patches(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Get the stored patches from the processed AnnData object and reshape them to the size specified in self.patch_size.
        The new patches will be stored in adata.obsm[f'patches_scale_{self.patch_scale}'] (overwriting the old ones).

        Args:
            processed_adata (ad.AnnData): The processed AnnData object. Must have the patches stored in adata.obsm[f'patches_scale_{self.patch_scale}'].

        Returns:
            ad.AnnData: The processed AnnData object with the new patches stored in adata.obsm[f'patches_scale_{self.patch_scale}']. Note that
                        they are flattened.
        """
        print(f'Reshaping patches to size {self.patch_size} X {self.patch_size}')
        
        start = time()
        # Get the original patches shape
        patches_shape = [int(np.sqrt(adata.obsm[f'patches_scale_{self.patch_scale}'].shape[1]//3))]*2 + [3]
        
        # Get the original patches
        original_patches = adata.obsm[f'patches_scale_{self.patch_scale}']
        
        # Reshape all the patches to the original shape
        all_images = original_patches.reshape((-1, patches_shape[0], patches_shape[1], patches_shape[2]))
        pil_images = [Image.fromarray(all_images[i, :, :, :]) for i in range(all_images.shape[0])]          # Turn all images to PIL
        resized_images = [im.resize((self.patch_size, self.patch_size)) for im in pil_images]               # Resize PIL images
        np_images = [np.array(im).flatten() for im in resized_images]                                       # Flatten images in numpy
        flat_matrix = np.vstack(tuple(np_images))                                                           # Add all images to matrix

        # Modify patches in original data
        adata.obsm[f'patches_scale_{self.patch_scale}'] = flat_matrix
        end = time()

        # Print time that took to reshape patches
        print(f'Reshape took {round(end - start, 3)} s')

        return adata

    def load_or_compute_adata(self) -> ad.AnnData:
        """
        This function does tha main data pipeline. It will first check if the processed data exists in the dataset_path. If it does not exist,
        then it will compute it and save it. If it does exist, then it will load it and return it. If it is in the compute mode, then it will
        also save quality control plots.

        Returns:
            ad.AnnData: The processed AnnData object ready to be used for training.
        """

        # If processed data does not exist, then compute and save it 
        if (not os.path.exists(os.path.join(self.dataset_path, f'adata.h5ad'))) or self.force_compute:
            
            print('Computing main adata file from downloaded raw data...')
            collection_raw = self.reader_class.get_adata_collection()
            collection_filtered = self.filter_dataset(collection_raw)
            collection_processed = self.process_dataset(collection_filtered)
            collection_reshaped = self.reshape_patches(collection_processed)

            # Save the processed data
            os.makedirs(self.dataset_path, exist_ok=True)
            collection_reshaped.write(os.path.join(self.dataset_path, f'adata.h5ad'))

            # QC plotting
            self.plot_tests(collection_reshaped, collection_raw)

        else:
            print(f'Loading main adata file from disk ({os.path.join(self.dataset_path, f"adata.h5ad")})...')
            # If the file already exists, load it
            collection_reshaped = ad.read(os.path.join(self.dataset_path, f'adata.h5ad'))

            print('The loaded adata object looks like this:')
            print(collection_reshaped)

        return collection_reshaped        
    
    # TODO: Update documentation
    def plot_tests(self, processed_adata: ad.AnnData, raw_adata: ad.AnnData)->None:
        """
        This function calls all the plotting functions in the class to create 6 quality control plots to check if the processing step of
        the dataset is performed correctly. The results are saved in dataset_logs folder and indexed by date and time. A dictionary
        in json format with all the dataset parameters is saved in the same log folder for reproducibility. Finally, a txt with the names of the
        genes used in processed adata is also saved in the folder.
        """

        ### Define function to get an adata list of plotting slides

        def get_plotting_slides_adata(self, collection: ad.AnnData, slide_list: str) -> list:
            """
            This function receives a string with a list of slides separated by commas and returns a list of anndata objects with
            the specified slides taken from the collection parameter. 

            Args:
                slide_list (str): String with a list of slides separated by commas.

            Returns:
                list: List of anndata objects with the specified slides.
            """
            # Get possible slide names to choose from
            slide_names = (collection.obs['slide_id']).unique()
            
            if slide_list == 'None':
                # Choose 4 random slides if possible (if there are less than 4 slides, choose all)
                if len(slide_names) < 4:
                    chosen_slides = slide_names
                else:
                    chosen_slides = np.random.choice(slide_names, size=4, replace=False)
            else:
                # Get the slides from the parameter dictionary
                chosen_slides = slide_list.split(',')
                assert len(chosen_slides) <= 4, 'You must specify at most 4 plotting slides separated by commas.'
                assert all([slide in slide_names for slide in chosen_slides]), 'Some of the plotting slides you specified are not in the dataset.'

            # Get the slides from the collection
            s_adata_list = [self.get_slide_from_collection(collection,  slide) for slide in chosen_slides]

            # Return the slides
            return s_adata_list
    
        ### Define all plotting functions

        def plot_histograms(self, processed_adata: ad.AnnData, raw_adata: ad.AnnData, path: str) -> None:
            """
            This function plots a figure that analyses the effect of the filtering over the data.
            The first row corresponds to the raw data (which has patches and excludes constant genes) and the second row
            plots the filtered and processed data. Histograms of total:
                
                1. Counts per cell
                2. Cells with expression
                3. Total counts per gene
                4. Moran I statistics (only in processed data)
            
            are generated. The plot is saved in the specified path.
            Cell filtering histograms are in red, gene filtering histograms are in blue and autocorrelation filtering histograms are in green.

            Args:
                self (SpatialDataset): Dataset object.
                processed_adata (ad.AnnData): Processed and filtered data ready to use by the model.
                raw_adata (ad.AnnData): Loaded data from .h5ad file that is not filtered but has patch information.
                path (str): Path to save histogram plot.
            """

            ### Define function to get the expression fraction
            def get_exp_frac(adata: ad.AnnData) -> ad.AnnData:
                """
                This function computes the expression fraction for each gene in the dataset. Internally it gets the
                expression fraction for each slide and then takes the minimum across all the slides.
                """
                # Get the unique slide ids
                slide_ids = adata.obs['slide_id'].unique()

                # Define zeros matrix of shape (n_genes, n_slides)
                exp_frac = np.zeros((adata.n_vars, len(slide_ids)))

                # Iterate over the slide ids
                for i, slide_id in enumerate(slide_ids):
                    # Get current slide adata
                    slide_adata = adata[adata.obs['slide_id'] == slide_id, :]
                    # Get current slide expression fraction
                    curr_exp_frac = np.squeeze(np.asarray((slide_adata.X > 0).sum(axis=0) / slide_adata.n_obs))
                    # Add current slide expression fraction to the matrix
                    exp_frac[:, i] = curr_exp_frac
                
                # Compute the minimum expression fraction for each gene across all the slides
                min_exp_frac = np.min(exp_frac, axis=1)

                # Add the minimum expression fraction to the var dataframe of the slide collection
                adata.var['exp_frac'] = min_exp_frac

                # Return the adata
                return adata

            # Compute qc metrics for raw and processed data in order to have total counts updated
            sc.pp.calculate_qc_metrics(raw_adata, inplace=True, log1p=False, percent_top=None)
            sc.pp.calculate_qc_metrics(processed_adata, inplace=True, log1p=False, percent_top=None, layer='counts')
            
            # Compute the expression fraction of the raw_adata
            raw_adata = get_exp_frac(raw_adata)

            # Create figures
            fig, ax = plt.subplots(nrows=2, ncols=5)
            fig.set_size_inches(18.75, 5)

            bin_num = 50

            # Plot histogram of the number of counts that each cell has
            raw_adata.obs['total_counts'].hist(ax=ax[0,0], bins=bin_num, grid=False, color='k')
            processed_adata.obs['total_counts'].hist(ax=ax[1,0], bins=bin_num, grid=False, color='darkred')

            # Plot histogram of the expression fraction of each gene
            raw_adata.var['exp_frac'].plot(kind='hist', ax=ax[0,1], bins=bin_num, grid=False, color='k', logy=True)
            processed_adata.var['exp_frac'].plot(kind = 'hist', ax=ax[1,1], bins=bin_num, grid=False, color='darkcyan', logy=True)

            # Plot histogram of the number of cells that express a given gene
            raw_adata.var['n_cells_by_counts'].plot(kind='hist', ax=ax[0,2], bins=bin_num, grid=False, color='k', logy=True)
            processed_adata.var['n_cells_by_counts'].plot(kind = 'hist', ax=ax[1,2], bins=bin_num, grid=False, color='darkcyan', logy=True)
            
            # Plot histogram of the number of total counts per gene
            raw_adata.var['total_counts'].plot(kind='hist', ax=ax[0,3], bins=bin_num, grid=False, color='k', logy=True)
            processed_adata.var['total_counts'].plot(kind = 'hist', ax=ax[1,3], bins=bin_num, grid=False, color='darkcyan', logy=True)
            
            # Plot histogram of the MoranI statistic per gene
            # raw_adata.var['moranI'].plot(kind='hist', ax=ax[0,4], bins=bin_num, grid=False, color='k', logy=True)
            processed_adata.var['d_log1p_moran'].plot(kind = 'hist', ax=ax[1,4], bins=bin_num, grid=False, color='darkgreen', logy=True)

            # Lists to format axes
            tit_list = ['Raw: Total counts',        'Raw: Expression fraction',         'Raw: Cells with expression',       'Raw: Total gene counts',       'Raw: MoranI statistic',
                        'Processed: Total counts',  'Processed: Expression fraction',   'Processed: Cells with expression', 'Processed: Total gene counts', 'Processed: MoranI statistic']
            x_lab_list = ['Total counts', 'Expression fraction', 'Cells with expression', 'Total counts', 'MoranI statistic']*2
            y_lab_list = ['# of cells', '# of genes', '# of genes', '# of genes', '# of genes']*2

            # Format axes
            for i, axis in enumerate(ax.flatten()):
                # Not show moran in raw data because it has no sense to compute it
                if i == 4:
                    # Delete frame 
                    axis.axis('off')
                    continue
                axis.set_title(tit_list[i])
                axis.set_xlabel(x_lab_list[i])
                axis.set_ylabel(y_lab_list[i])
                axis.spines[['right', 'top']].set_visible(False)

            # Shared x axes between plots
            ax[1,0].sharex(ax[0,0])
            ax[1,1].sharex(ax[0,1])
            ax[1,2].sharex(ax[0,2])
            ax[1,3].sharex(ax[0,3])
            ax[1,4].sharex(ax[0,4])

            # Shared y axes between
            ax[1,0].sharey(ax[0,0])
            
            fig.tight_layout()
            fig.savefig(path, dpi=300)
            plt.close()

        def plot_random_patches(self, processed_adata: ad.AnnData, path: str) -> None:
            """
            This function gets 16 flat random patches (with the specified dims) from the processed adata objects. It
            reshapes them to a bidimensional form and shows them. The plot is saved to the specified path.

            Args:
                self (SpatialDataset): Dataset object.
                processed_adata (ad.AnnData): Processed and filtered data ready to use by the model.
                path (str): Path to save the image.
            """
            # Get the flat patches from the dataset
            flat_patches = processed_adata.obsm[f'patches_scale_{self.patch_scale}']
            # Reshape the patches for them to have image form
            patches = flat_patches.reshape((-1, self.patch_size, self.patch_size, 3))
            # Choose 16 random patches
            chosen = np.random.randint(low=0, high=patches.shape[0], size=16)
            # Get plotting patches
            plotting_patches = patches[chosen, :, :, :]

            # Declare image
            im, ax = plt.subplots(nrows=4, ncols=4)

            # Cycle over each random patch
            for i, ax in enumerate(ax.reshape(-1)):
                ax.imshow(plotting_patches[i, :, :, :])
                ax.axis('off')
            
            # Set figure formatting
            im.suptitle(f'Random patches from {self.dataset}')
            plt.tight_layout()
            im.savefig(path, dpi=300)
            plt.close()

        def visualize_moran_filtering(self, processed_adata: ad.AnnData, from_layer: str, path: str, top: bool = True) -> None:
            """
            This function visualizes the spatial expression of the 4 most and least auto-correlated genes in processed_adata.
            The title of each subplot shows the value of the moran I statistic for a given gene. The plot is saved to the specified
            path. This plot uses the slide list in string format in self.param_dict['plotting_slides']to plot these specific observations.
            If no list is provided (self.param_dict['plotting_slides']=='None'), 4 random slides are chosen. 

            Args:
                self (SpatialDataset): Dataset object.
                processed_adata (ad.AnnData): Processed and filtered data ready to use by the model.
                from_layer (str): Layer of the adata object to use for plotting.
                path (str): Path to save the generated image.
                top (bool, optional): If True, the top 4 most auto-correlated genes are visualized. If False, the top 4 least
                                    auto-correlated genes are visualized. Defaults to True.
            """
            
            plotting_key = from_layer

            # Get the slides to visualize in adata format
            s_adata_list = get_plotting_slides_adata(self, processed_adata, self.param_dict['plotting_slides'])

            # Get te top 4 most or least auto-correlated genes in processed data depending on the value of top
            # NOTE: The selection of genes is done in the complete collection of slides, not in the specified slides
            moran_key = 'd_log1p_moran'
            if top:
                selected_table = processed_adata.var.nlargest(4, columns=moran_key)
            else:
                selected_table = processed_adata.var.nsmallest(4, columns=moran_key)

            # Declare figure
            if self.dataset == "stnet_dataset":
                fig, ax = plt.subplots(nrows=4, ncols=4)
                fig.set_size_inches(14, 13)
            else:
                fig, ax = plt.subplots(nrows=4, ncols=2)
                fig.set_size_inches(10, 13)

            # Cycle over slides
            for i in range(len(selected_table)):

                # Get min and max of the selected gene in the slides
                gene_min = min([dat[:, selected_table.index[i]].layers[plotting_key].min() for dat in s_adata_list])
                gene_max = max([dat[:, selected_table.index[i]].layers[plotting_key].max() for dat in s_adata_list])

                # Define color normalization
                norm = matplotlib.colors.Normalize(vmin=gene_min, vmax=gene_max)

                for j in range(len(s_adata_list)):
                    
                    # Define bool to only plot the colorbar in the last column
                    cbar = True if j==(len(s_adata_list)-1) else False

                    # Plot selected genes in the specified slides
                    sq.pl.spatial_scatter(s_adata_list[j], color=[selected_table.index[i]], layer= plotting_key, ax=ax[i,j], cmap='jet', norm=norm, colorbar=cbar)
                    
                    # Set slide name
                    if i==0:
                        ax[i,j].set_title(f'{self.param_dict["plotting_slides"].split(",")[j]}', fontsize=15)
                    else:
                        ax[i,j].set_title('')
                    
                    # Set gene name and moran I value
                    if j==0:
                        ax[i,j].set_ylabel(f'{selected_table.index[i]}: $I = {selected_table[moran_key][i].round(3)}$', fontsize=13)
                    else:
                        ax[i,j].set_ylabel('')

            # Format figure
            for axis in ax.flatten():
                axis.set_xlabel('')
                # Turn off all spines
                axis.spines['top'].set_visible(False)
                axis.spines['right'].set_visible(False)
                axis.spines['bottom'].set_visible(False)
                axis.spines['left'].set_visible(False)

            # Define title
            tit_str = 'most (top)' if top else 'least (bottom)'

            fig.suptitle(f'Top 4 {tit_str} auto-correlated genes in processed data', fontsize=20)
            fig.tight_layout()
            # Save plot 
            fig.savefig(path, dpi=300)
            plt.close()

        def visualize_gene_expression(self, processed_adata: ad.AnnData, from_layer: str, path: str) -> None:
            """
            This function selects the genes specified in self.param_dict['plotting_genes'] and self.param_dict['plotting_slides']
            to plot gene expression for the specified genes in the specified slides. If either of them is 'None', then the method
            chooses randomly (4 genes or 4 slides in the stnet_dataset or 2 slides in visium datasets). The data is plotted from
            the .layers[from_layer] expression matrix

            Args:
                self (SpatialDataset): Dataset object.
                processed_adata (ad.AnnData): The processed adata with the filtered patient collection
                from_layer (str): The key to the layer of the data to plot.
                path (str): Path to save the image.
            """

            # Get the slides to visualize in adata format
            s_adata_list = get_plotting_slides_adata(self, processed_adata, self.param_dict['plotting_slides'])

            # Define gene list
            gene_list = self.param_dict['plotting_genes'].split(',')

            # Try to get the specified genes otherwise choose randomly
            try:
                gene_table = processed_adata[:, gene_list].var
            except:
                print('Could not find all the specified plotting genes, choosing randomly')
                gene_list = np.random.choice(processed_adata.var_names, size=4, replace=False)
                gene_table = processed_adata[:, gene_list].var
            
            # Declare figure
            if self.dataset == "stnet_dataset":
                fig, ax = plt.subplots(nrows=4, ncols=4)
                fig.set_size_inches(14, 13)
            else:
                fig, ax = plt.subplots(nrows=4, ncols=2)
                fig.set_size_inches(10, 13)

            # Cycle over slides
            for i in range(len(gene_table)):

                # Get min and max of the selected gene in the slides
                gene_min = min([dat[:, gene_table.index[i]].layers[from_layer].min() for dat in s_adata_list])
                gene_max = max([dat[:, gene_table.index[i]].layers[from_layer].max() for dat in s_adata_list])

                # Define color normalization
                norm = matplotlib.colors.Normalize(vmin=gene_min, vmax=gene_max)

                for j in range(len(s_adata_list)):

                    # Define bool to only plot the colorbar in the last column
                    cbar = True if j==(len(s_adata_list)-1) else False
                    
                    # Plot selected genes in the specified slides
                    sq.pl.spatial_scatter(s_adata_list[j], layer=from_layer, color=[gene_table.index[i]], ax=ax[i,j], cmap='jet', norm=norm, colorbar=cbar)
                    
                    # Set slide name
                    if i==0:
                        ax[i,j].set_title(f'{self.param_dict["plotting_slides"].split(",")[j]}', fontsize=15)
                    else:
                        ax[i,j].set_title('')
                    
                    # Set gene name with moran I value 
                    if j==0:
                        moran_key = 'd_log1p_moran'
                        ax[i,j].set_ylabel(f'{gene_table.index[i]}: $I = {gene_table[moran_key][i].round(3)}$', fontsize=13)
                    else:
                        ax[i,j].set_ylabel('')
            
            # Format figure
            for axis in ax.flatten():
                axis.set_xlabel('')
                # Turn off all spines
                axis.spines['top'].set_visible(False)
                axis.spines['right'].set_visible(False)
                axis.spines['bottom'].set_visible(False)
                axis.spines['left'].set_visible(False)
            
            fig.suptitle('Gene expression in processed data', fontsize=20)
            fig.tight_layout()
            # Save plot
            fig.savefig(path, dpi=300)
            plt.close()

        def plot_clusters(self, processed_adata: ad.AnnData, from_layer: str, path: str) -> None:
            """
            This function generates a plot that visualizes Leiden clusters spatially in the slides in self.param_dict['plotting_slides'].
            The slides can be specified in self.param_dict['plotting_slides'] or chosen randomly.
            
            It plots:
                1. The spatial distribution of the Leiden clusters in the slides.
                2. UMAP embeddings of each slide colored by Leiden clusters.
                3. General UMAP embedding of the complete dataset colored by Leiden clusters and the batch correction key.
                4. PCA embeddings of the complete dataset colored by the batch correction key.

            Args:
                self (SpatialDataset): Dataset object.
                processed_adata (ad.AnnData): Processed and filtered data ready to use by the model.
                from_layer (str): The key in adata.layers where the expression matrix is stored.
                path (str): Path to save the image.
            """

            ### Define function to get dimensionality reductions depending on the layer
            def compute_dim_red(adata: ad.AnnData, from_layer: str) -> ad.AnnData:
                """
                Simple wrapper around sc.pp.pca, sc.pp.neighbors, sc.tl.umap and sc.tl.leiden to compute the embeddings and cluster the data.
                Everything will be computed using the expression matrix stored in adata.layers[from_layer]. 

                Args:
                    adata (ad.AnnData): The AnnData object to transform. Must have expression values in adata.layers[from_layer].
                    from_layer (str): The key in adata.layers where the expression matrix is stored.

                Returns:
                    ad.AnnData: The transformed AnnData object with the embeddings and clusters.
                """
                
                # Start the timer
                # start = time()
                # print(f'Computing embeddings and clusters using data of layer {from_layer}...')
                
                # Set the key layer as the main expression matrix
                adata_copy = adata.copy()
                adata_copy.X = adata_copy.layers[from_layer]
                

                # Compute the embeddings and clusters
                sc.pp.pca(adata_copy, random_state=42)
                sc.pp.neighbors(adata_copy, random_state=42)
                sc.tl.umap(adata_copy, random_state=42)
                sc.tl.leiden(adata_copy, key_added="cluster", random_state=42)
                
                # Restore the original expression matrix as counts layer
                adata_copy.X = adata_copy.layers['counts']

                # Print the time it took to compute the embeddings and clusters
                # print(f'Embeddings and clusters computed in {time() - start:.2f} seconds')

                # Return the adapted AnnData object
                return adata_copy

            # Update the adata object with the embeddings and clusters
            updated_adata = compute_dim_red(processed_adata, from_layer)

            # Get the slides to visualize in adata format
            s_adata_list = get_plotting_slides_adata(self, updated_adata, self.param_dict['plotting_slides'])

            # Define dictionary from cluster to color
            clusters = updated_adata.obs['cluster'].unique()
            # Sort clusters
            clusters = np.sort([int(cl) for cl in clusters])
            clusters = [str(cl) for cl in clusters]
            # Define color palette
            colors = sns.color_palette('hls', len(clusters))
            palette = dict(zip(clusters, colors))
            gray_palette = dict(zip(clusters, ['gray']*len(clusters)))

            # Declare figure
            fig = plt.figure(layout="constrained")
            gs0 = fig.add_gridspec(1, 2)
            gs00 = gs0[0].subgridspec(4, 2)
            gs01 = gs0[1].subgridspec(3, 1)

            fig.set_size_inches(15,14)

            # Cycle over slides
            for i in range(len(s_adata_list)):
                
                curr_clusters = s_adata_list[i].obs['cluster'].unique()
                # Sort clusters
                curr_clusters = np.sort([int(cl) for cl in curr_clusters])
                curr_clusters = [str(cl) for cl in curr_clusters]
                # # Define color palette
                spatial_colors = matplotlib.colors.ListedColormap([palette[x] for x in curr_clusters])
                
                # Get ax for spatial plot and UMAP plot
                spatial_ax = fig.add_subplot(gs00[i, 0])
                umap_ax = fig.add_subplot(gs00[i, 1])

                # Plot cluster colors in spatial space
                sq.pl.spatial_scatter(s_adata_list[i], color=['cluster'], ax=spatial_ax, palette=spatial_colors)

                spatial_ax.get_legend().remove()
                spatial_ax.set_title('Spatial', fontsize=18)
                spatial_ax.set_ylabel(f'{self.param_dict["plotting_slides"].split(",")[i]}', fontsize=12)
                spatial_ax.set_xlabel('')
                # Turn off all spines
                spatial_ax.spines['top'].set_visible(False)
                spatial_ax.spines['right'].set_visible(False)
                spatial_ax.spines['bottom'].set_visible(False)
                spatial_ax.spines['left'].set_visible(False)
                

                # Plot cluster colors in UMAP space for slide and all collection
                sc.pl.umap(updated_adata, layer=from_layer, color=['cluster'], ax=umap_ax, frameon=False, palette=gray_palette, s=10, cmap=None, alpha=0.2)
                umap_ax.get_legend().remove()
                sc.pl.umap(s_adata_list[i], layer=from_layer, color=['cluster'], ax=umap_ax, frameon=False, palette=palette, s=10, cmap=None)
                umap_ax.get_legend().remove()
                umap_ax.set_title('UMAP', fontsize=18)
                
            # Get axes for leiden clusters, patient and cancer types
            leiden_ax = fig.add_subplot(gs01[0])
            patient_ax = fig.add_subplot(gs01[1])
            pca_ax = fig.add_subplot(gs01[2])

            # Plot leiden clusters in UMAP space
            sc.pl.umap(updated_adata, color=['cluster'], ax=leiden_ax, frameon=False, palette=palette, s=10, cmap=None)
            leiden_ax.get_legend().set_title('Leiden Clusters')
            leiden_ax.get_legend().get_title().set_fontsize(15)
            leiden_ax.set_title('UMAP & Leiden Clusters', fontsize=18)

            # Plot batch_key in UMAP space
            sc.pl.umap(updated_adata, color=[self.param_dict['combat_key']], ax=patient_ax, frameon=False, palette='tab20', s=10, cmap=None)
            patient_ax.get_legend().set_title(self.param_dict['combat_key'])
            patient_ax.get_legend().get_title().set_fontsize(15)
            patient_ax.set_title(f"UMAP & {self.param_dict['combat_key']}", fontsize=18)

            # Plot cancer types in UMAP space
            sc.pl.pca(updated_adata, color=[self.param_dict['combat_key']], ax=pca_ax, frameon=False, palette='tab20', s=10, cmap=None)
            pca_ax.get_legend().set_title(self.param_dict['combat_key'])
            pca_ax.get_legend().get_title().set_fontsize(15)
            pca_ax.set_title(f'PCA & {self.param_dict["combat_key"]}', fontsize=18)
            
            # Format figure and save
            fig.suptitle(f'Cluster visualization for {self.dataset} in layer {from_layer}', fontsize=25)
            # fig.tight_layout()
            fig.savefig(path, dpi=300)
            plt.close(fig)

        def plot_cell_filtering(self, path: str) -> None:
            """
            This function plots the total counts per cell prior and after filtering overlaid with the tissue.
            This plotting is done for the chosem slides that can be specified in the self.param_dict['plotting_slides'] or 
            chosen randomly if the parameter is 'None'. The data does not reflect normalization or gene filtering
            but serves to appreciate the spatial distribution of the cells that passed filtering. 

            Args:
                self (SpatialDataset): Dataset object.
                path (str): Path to save the image.
            """

            # Get raw and processed adata for the slides to plot
            raw_s_adata_list = get_plotting_slides_adata(self, processed_adata, self.param_dict['plotting_slides'])
            processed_s_adata_list = get_plotting_slides_adata(self, processed_adata, self.param_dict['plotting_slides'])

            # Declare figure
            fig, ax = plt.subplots(ncols=len(raw_s_adata_list), nrows=2)
            if self.dataset == "stnet_dataset":
                fig.set_size_inches(15, 6)
            else:
                fig.set_size_inches(10, 6)
            
            for i in range(len(raw_s_adata_list)):
                # Plot spatially total counts of raw and processed adata
                sq.pl.spatial_scatter(raw_s_adata_list[i], color=['total_counts'], ax=ax[0, i], cmap='jet')
                sq.pl.spatial_scatter(processed_s_adata_list[i], color=['total_counts'], ax=ax[1, i], cmap='jet')

                if i == 0:
                    ax[0,i].set_ylabel('Raw', fontsize='x-large')
                    ax[1,i].set_ylabel('Processed', fontsize='x-large')
                else:
                    ax[0,i].set_ylabel('')
                    ax[1,i].set_ylabel('')
                
                # Set title
                ax[0,i].set_title(f'{self.param_dict["plotting_slides"].split(",")[i]}', fontsize='x-large')
                ax[1,i].set_title('')

            # Format axes
            for i, axis in enumerate(ax.flatten()):
                axis.set_xlabel('')
                axis.collections[-1].colorbar.set_label('Total counts', fontsize='large')
                # Remove all spines
                axis.spines['top'].set_visible(False)
                axis.spines['right'].set_visible(False)
                axis.spines['bottom'].set_visible(False)
                axis.spines['left'].set_visible(False)


            # Format figure and save
            fig.suptitle(f'Valid Cells Visualization for {self.dataset}', fontsize=20)
            fig.tight_layout()
            fig.savefig(path, dpi=300)
            plt.close(fig)

        def plot_mean_std(self, processed_adata: ad.AnnData, raw_adata: ad.AnnData, path: str) -> None:
            """
            This function plots a scatter of mean and standard deviation of genes present in raw_adata (black) and all the layers with non-zero
            mean in processed_adata. It is used to see the effect of filtering and processing in the genes. The plot is saved to the specified path.

            Args:
                self (SpatialDataset): Dataset object.
                processed_adata (ad.AnnData): Processed and filtered data ready to use by the model.
                raw_adata (ad.AnnData): Data loaded data from .h5ad file that is not filtered but has patch information.
                path (str): Path to save the image.
            """
            # Copy raw data to auxiliary data
            aux_raw_adata = raw_adata.copy()

            # Normalize and log transform aux_raw_adata
            sc.pp.normalize_total(aux_raw_adata, inplace=True)
            sc.pp.log1p(aux_raw_adata)

            # Get means and stds from raw data
            raw_mean = aux_raw_adata.to_df().mean(axis=0)
            raw_std = aux_raw_adata.to_df().std(axis=0)

            # Define list of layers to plot
            layers = ['log1p', 'd_log1p', 'c_log1p', 'c_d_log1p']

            plt.figure()
            plt.scatter(raw_mean, raw_std, s=1, c='k', label='Raw data')
            for layer in layers:
                # Get means and stds from processed data
                pro_mean = processed_adata.to_df(layer=layer).mean(axis=0)
                pro_std = processed_adata.to_df(layer=layer).std(axis=0)
                plt.scatter(pro_mean, pro_std, s=1, label=f'{layer} data')
            plt.xlabel('Mean $Log(x+1)$')
            plt.ylabel('Std $Log(x+1)$')
            plt.legend(loc='best')
            plt.title(f'Mean Std plot {self.dataset}')
            plt.gca().spines[['right', 'top']].set_visible(False)

            plt.tight_layout()
            plt.savefig(path, dpi=300)
            plt.close()

        print('Started quality control plotting')
        start = time()

        # Define directory path to save data
        save_path = os.path.join(self.dataset_path, 'qc_plots')
        os.makedirs(save_path, exist_ok=True)

        # Assure that the plotting genes are in the data and if not, set random plotting genes
        if not all([gene in processed_adata.var_names for gene in self.param_dict['plotting_genes'].split(',')]):
            self.param_dict['plotting_genes'] = ','.join(np.random.choice(processed_adata.var_names, 4))
            print(f'Plotting genes not in data. Setting random plotting genes: {self.param_dict["plotting_genes"]}')

        # Make plot of filtering histograms
        print('Started filtering histograms plotting')
        plot_histograms(self, processed_adata, raw_adata, os.path.join(save_path, 'filtering_histograms.png'))

        # Make plot of random patches
        print('Started random patches plotting')
        plot_random_patches(self, processed_adata, os.path.join(save_path, 'random_patches.png'))

        # Define moran and cluster available layers
        moran_cluster_layers = ['log1p', 'd_log1p', 'c_log1p', 'c_d_log1p', 'deltas', 'd_deltas', 'c_deltas', 'c_d_deltas']

        # Create save paths fot top and bottom moran genes
        os.makedirs(os.path.join(save_path, 'top_moran_genes'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'bottom_moran_genes'), exist_ok=True)
        print('Started moran filtering plotting')
        # Plot moran filtering
        for lay in tqdm(moran_cluster_layers):
            # Make plot of 4 most moran genes and 4 less moran genes (in the chosen slides)
            visualize_moran_filtering(self, processed_adata, from_layer=lay, path = os.path.join(save_path, 'top_moran_genes', f'{lay}.png'), top = True)
            visualize_moran_filtering(self, processed_adata, from_layer=lay, path = os.path.join(save_path, 'bottom_moran_genes', f'{lay}.png'), top = False)
        

        # Create save paths for cluster plots
        os.makedirs(os.path.join(save_path, 'cluster_plots'), exist_ok=True)
        print('Started cluster plotting')
        # Plot cluster graphs
        for lay in tqdm(moran_cluster_layers):
            plot_clusters(self, processed_adata, from_layer=lay, path=os.path.join(save_path, 'cluster_plots', f'{lay}.png'))
        
        # Define expression layers
        expression_layers = ['counts', 'tpm', 'log1p', 'd_log1p', 'c_log1p', 'c_d_log1p', 'deltas', 'd_deltas', 'c_deltas', 'c_d_deltas']
        os.makedirs(os.path.join(save_path, 'expression_plots'), exist_ok=True)
        print('Started gene expression plotting')
        # Plot of gene expression in the chosen slides for the 4 chosen genes
        for lay in tqdm(expression_layers):
            visualize_gene_expression(self, processed_adata, from_layer=lay, path=os.path.join(save_path,'expression_plots', f'{lay}.png'))

        # Make plot of active cells filtering
        print('Started cell filtering plotting')
        plot_cell_filtering(self, os.path.join(save_path,'cell_filtering.png'))

        # Make plot of mean vs std per gene must be programmed manually.
        print('Started mean vs std plotting')
        plot_mean_std(self, processed_adata, raw_adata, os.path.join(save_path, 'mean_std_scatter.png'))
        
        # Print the time that took to plot quality control
        end = time()
        print(f'Quality control plotting took {round(end-start, 2)}s')
        print(f'Images saved in {save_path}')

    def compute_patches_embeddings_and_predictions(self, backbone: str ='densenet', model_path:str="best_stnet.pt", preds: bool=True) -> None:
            
            # Define a cuda device if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model = models.ImageEncoder(backbone=backbone, use_pretrained=True, latent_dim=self.adata.n_vars)
            saved_model = torch.load(model_path)
            model.load_state_dict(saved_model)
            
            if backbone == 'resnet':
                weights = tmodels.ResNet18_Weights.DEFAULT
                if not preds:
                    model.encoder.fc = nn.Identity()
            elif backbone == 'ConvNeXt':
                weights = tmodels.ConvNeXt_Tiny_Weights.DEFAULT
                if not preds:
                    model.encoder.classifier[2] = nn.Identity()
            elif backbone == 'EfficientNetV2':
                weights = tmodels.EfficientNet_V2_S_Weights.DEFAULT 
                if not preds:
                    model.encoder.classifier[1] = nn.Identity()
            elif backbone == 'InceptionV3':
                weights = tmodels.Inception_V3_Weights.DEFAULT
                if not preds:
                    model.encoder.fc = nn.Identity()
            elif backbone == "MaxVit":
                weights = tmodels.MaxVit_T_Weights.DEFAULT
                if not preds:
                    model.encoder.classifier[5] = nn.Identity()
            elif backbone == "MobileNetV3":
                weights = tmodels.MobileNet_V3_Small_Weights.DEFAULT
                if not preds:
                    model.encoder.classifier[3] = nn.Identity()
            elif backbone == "ResNetXt":
                weights = tmodels.ResNeXt50_32X4D_Weights.DEFAULT
                if not preds:
                    model.encoder.fc = nn.Identity()
            elif backbone == "ShuffleNetV2":
                weights = tmodels.ShuffleNet_V2_X0_5_Weights.DEFAULT
                if not preds:
                    model.encoder.fc = nn.Identity()
            elif backbone == "ViT":
                weights = tmodels.ViT_B_16_Weights.DEFAULT
                if not preds:
                    model.encoder.heads.head = nn.Identity()
            elif backbone == "WideResnet":
                weights = tmodels.Wide_ResNet50_2_Weights.DEFAULT
                if not preds:
                    model.encoder.fc = nn.Identity()
            elif backbone == 'densenet':
                weights = tmodels.DenseNet121_Weights.DEFAULT
                if not preds:
                    model.encoder.classifier = nn.Identity() 
            elif backbone == 'swin':
                weights = tmodels.Swin_T_Weights.DEFAULT
                if not preds:
                    model.encoder.head = nn.Identity()
            else:
                raise ValueError(f'Backbone {backbone} not supported')

            model.to(device)
            model.eval()

            preprocess = weights.transforms(antialias=True)

            # Get the patches
            flat_patches = self.adata.obsm[f'patches_scale_{self.patch_scale}']

            # Reshape all the patches to the original shape
            all_patches = flat_patches.reshape((-1, self.patch_size, self.patch_size, 3))
            torch_patches = torch.from_numpy(all_patches).permute(0, 3, 1, 2).float()    # Turn all patches to torch
            rescaled_patches = torch_patches / 255                                       # Rescale patches to [0, 1]
            processed_patches = preprocess(rescaled_patches)                             # Preprocess patches
            
            # Create a dataloader
            dataloader = DataLoader(processed_patches, batch_size=256, shuffle=False, num_workers=4)

            # Declare lists to store the embeddings or predictions
            outputs = []

            with torch.no_grad():
                if preds:
                    desc = 'Getting predictions'
                else:
                    desc = 'Getting embeddings'
                for batch in tqdm(dataloader, desc=desc):
                    batch = batch.to(device)                    # Send batch to device                
                    batch_output = model(batch)                 # Get embeddings or predictions
                    outputs.append(batch_output)                # Append to list


            # Concatenate all embeddings or predictions
            outputs = torch.cat(outputs, dim=0)
        
            # Pass embeddings or predictions to cpu and add to self.data.obsm
            if preds:
                self.adata.obsm['predictions'] = outputs.cpu().numpy()
            else:
                self.adata.obsm['embeddings'] = outputs.cpu().numpy()

    def obtain_embeddings_resnet50(self):

        def extract(encoder, patches):
            return encoder(patches).view(-1,features)

        egn_transforms = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        resnet_encoder = tmodels.resnet50(True)
        features = resnet_encoder.fc.in_features
        modules = list(resnet_encoder.children())[:-1] # Encoder corresponds to ResNet50 without the fc layer
        resnet_encoder = torch.nn.Sequential(*modules)
        for p in resnet_encoder.parameters():
            p.requires_grad = False

        resnet_encoder = resnet_encoder.to(device)
        resnet_encoder.eval()

        # Get the patches
        flat_patches = self.adata.obsm[f'patches_scale_{self.patch_scale}']

        # Reshape all the patches to the original shape
        all_patches = flat_patches.reshape((-1, self.patch_size, self.patch_size, 3))
        torch_patches = torch.from_numpy(all_patches).permute(0, 3, 1, 2).float()    # Turn all patches to torch
        rescaled_patches = egn_transforms(torch_patches / 255)                       # Rescale patches to [0, 1]

        img_embedding = [extract(resnet_encoder, single_patch.unsqueeze(dim=0).to(device)) for single_patch in tqdm(rescaled_patches)]
        img_embedding = torch.cat(img_embedding).contiguous()             
        
        self.adata.obsm['resnet50_embeddings'] = img_embedding.cpu().numpy()
    
    def get_nn_images(self) -> None:

        def get_nn_dist_and_ids_images(query_adata: ad.AnnData, ref_adata: ad.AnnData) -> Tuple[pd.DataFrame, pd.DataFrame]:
            
            # Get embeddings from query and ref as torch tensors
            query_embeddings = torch.Tensor(query_adata.obsm['resnet50_embeddings'])
            ref_embeddings = torch.Tensor(ref_adata.obsm['resnet50_embeddings'])

            # Compute euclidean distances from query to ref
            query_ref_distances = torch.cdist(query_embeddings, ref_embeddings, p=2)

            # Get the sorted distances and indexes
            sorted_distances, sorted_indexes = torch.sort(query_ref_distances, dim=1)

            # Trim the sorted distances and indexes to 100 nearest neighbors and convert to numpy
            sorted_distances = sorted_distances[:, :100].numpy()
            sorted_indexes = sorted_indexes[:, :100].numpy()

            # Get index vector in numpy (just to avoid warnings)
            index_vector = ref_adata.obs.index.values

            # Get the ids of the 100 nearest neighbors
            sorted_ids = index_vector[sorted_indexes]
            
            # Make a dataframe with the distances and ids with the query index as index
            sorted_distances_df = pd.DataFrame(sorted_distances, index=query_adata.obs.index.values)
            sorted_ids_df = pd.DataFrame(sorted_ids, index=query_adata.obs.index.values)

            return sorted_distances_df, sorted_ids_df

        print('Getting image nearest neighbors...')
        start = time()

        # Define train subset (this is where val and test will look for nearest neighbors)
        train_subset = self.adata[self.adata.obs['split'] == 'train']
        val_subset = self.adata[self.adata.obs['split'] == 'val']
        test_subset = self.adata[self.adata.obs['split'] == 'test']
        
        # Use the get_nn_dist_and_ids function to get the distances and ids of the nearest neighbors
        val_train_distances_df, val_train_ids_df = get_nn_dist_and_ids_images(val_subset, train_subset)
        test_train_distances_df, test_train_ids_df = get_nn_dist_and_ids_images(test_subset, train_subset)

        # Now get the patients of the train set
        train_patients = train_subset.obs['patient'].unique()
        
        # Define list of dataframes to store the distances and ids from the train set to the train set
        train_train_distances_dfs = []
        train_train_ids_dfs = []

        # Cycle through train patients
        for patient in train_patients:

            # Get the train patient data
            patient_data = train_subset[train_subset.obs['patient'] == patient]
            # Get the train patient data that is not for the current patient
            other_patient_data = train_subset[train_subset.obs['patient'] != patient]

            # Apply the get_nn_dist_and_ids function to get the distances and ids of the nearest neighbors
            curr_patient_distances_df, curr_patient_ids_df = get_nn_dist_and_ids_images(patient_data, other_patient_data)
            
            # Append the dataframes to the list
            train_train_distances_dfs.append(curr_patient_distances_df)
            train_train_ids_dfs.append(curr_patient_ids_df)

        # Concatenate the dataframes
        train_train_distances_df = pd.concat(train_train_distances_dfs)
        train_train_ids_df = pd.concat(train_train_ids_dfs)

        # Concatenate train, val and test distances and ids
        all_distances_df = pd.concat([train_train_distances_df, val_train_distances_df, test_train_distances_df])
        all_ids_df = pd.concat([train_train_ids_df, val_train_ids_df, test_train_ids_df])

        # Reindex the dataframes
        all_distances_df = all_distances_df.reindex(self.adata.obs.index.values)
        all_ids_df = all_ids_df.reindex(self.adata.obs.index.values)
        
        # Add the dataframes to the obsm
        self.adata.obsm['image_nn_distances'] = all_distances_df
        self.adata.obsm['image_nn_ids'] = all_ids_df

        end = time()
        print(f'Finished getting image nearest neighbors in {end - start:.2f} seconds')

    def get_graphs(self, n_hops: int, layer: str) -> dict:
        """
        This function wraps the get_graphs_one_slide function to get the graphs for all the slides in the dataset. For details
        on the get_graphs_one_slide function see its docstring.

        Args:
            n_hops (int): The number of hops to compute each graph.
            layer (str): The layer of the graph to predict. Will be added as y to the graph.

        Returns:
            dict: A dictionary where the slide names are the keys and pytorch geometric graphs are values.
        """

        ### Define auxiliar functions ###

        def get_graphs_one_slide(self, adata: ad.AnnData, n_hops: int, layer: str, hex_geometry: bool) -> Tuple[dict,int]:
            """
            This function receives an AnnData object with a single slide and for each node computes the graph in an
            n_hops radius in a pytorch geometric format. It returns a dictionary where the patch names are the keys
            and a pytorch geometric graph for each one as values. NOTE: The first node of every graph is the center.

            Args:
                adata (ad.AnnData): The AnnData object with the slide data.
                n_hops (int): The number of hops to compute the graph.
                layer (str): The layer of the graph to predict. Will be added as y to the graph.
                hex_geometry (bool): Whether the slide has hexagonal geometry or not.

            Returns:
                Tuple(dict,int)
                dict: A dictionary where the patch names are the keys and pytorch geometric graph for each one as values.
                    NOTE: The first node of every graph is the center.
                int: Max absolute value of d pos in the slide                      
            """
            # Compute spatial_neighbors
            if hex_geometry:
                sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=6) # Hexagonal visium case
            else:
                sq.gr.spatial_neighbors(adata, coord_type='grid', n_neighs=8) # Grid STNet dataset case

            # Get the adjacency matrix
            adj_matrix = adata.obsp['spatial_connectivities']

            # Define power matrix
            power_matrix = adj_matrix.copy()
            # Define the output matrix
            output_matrix = adj_matrix.copy()

            # Iterate through the hops
            for i in range(n_hops-1):
                # Compute the next hop
                power_matrix = power_matrix * adj_matrix
                # Add the next hop to the output matrix
                output_matrix = output_matrix + power_matrix

            # Zero out the diagonal
            output_matrix.setdiag(0)
            # Threshold the matrix to 0 and 1
            output_matrix = output_matrix.astype(bool).astype(int)

            # Define dict from index to obs name
            index_to_obs = {i: obs for i, obs in enumerate(adata.obs.index.values)}

            # Define neighbors dicts (one with names and one with indexes)
            neighbors_dict_index = {}
            neighbors_dict_names = {}
            matrices_dict = {}

            # Iterate through the rows of the output matrix
            for i in range(output_matrix.shape[0]):
                # Get the non-zero elements of the row
                non_zero_elements = output_matrix[i].nonzero()[1]
                # Get the names of the neighbors
                non_zero_names = [index_to_obs[index] for index in non_zero_elements]
                # Add the neighbors to the neighbors dicts. NOTE: the first index is the query obs
                neighbors_dict_index[i] = [i] + list(non_zero_elements)
                neighbors_dict_names[index_to_obs[i]] = np.array([index_to_obs[i]] + non_zero_names)
                
                # Subset the matrix to the non-zero elements and store it in the matrices dict
                matrices_dict[index_to_obs[i]] = output_matrix[neighbors_dict_index[i], :][:, neighbors_dict_index[i]]

            
            ### Get pytorch geometric graphs ###
            patch_names = adata.obs.index.values                                                                        # Get global patch names
            layers_dict = {key: torch.from_numpy(adata.layers[key]).type(torch.float32) for key in adata.layers.keys()} # Get global layers
            patches = torch.from_numpy(adata.obsm[f'patches_scale_{self.patch_scale}'])                                 # Get global patches
            pos = torch.from_numpy(adata.obs[['array_row', 'array_col']].values)                                        # Get global positions

            # If embeddings and predictions are present in obsm, get them
            embeddings = torch.from_numpy(adata.obsm['embeddings']).type(torch.float32) if 'embeddings' in adata.obsm.keys() else None
            predictions = torch.from_numpy(adata.obsm['predictions']).type(torch.float32) if 'predictions' in adata.obsm.keys() else None

            # If layer contains delta then add a used_mean attribute to the graph
            used_mean = torch.from_numpy(self.adata.var[f'{layer}_avg_exp'.replace('deltas', 'log1p')].values).type(torch.float32) if 'deltas' in layer else None

            # Define the empty graph dict
            graph_dict = {}
            max_abs_d_pos=-1

            # Cycle over each obs
            for i in tqdm(range(len(neighbors_dict_index)), leave=False, position=1):
                central_node_name = index_to_obs[i]                                                 # Get the name of the central node
                curr_nodes_idx = torch.tensor(neighbors_dict_index[i])                              # Get the indexes of the nodes in the graph
                curr_adj_matrix = matrices_dict[central_node_name]                                  # Get the adjacency matrix of the graph (precomputed)
                curr_edge_index, curr_edge_attribute = from_scipy_sparse_matrix(curr_adj_matrix)    # Get the edge index and edge attribute of the graph
                curr_layers = {key: layers_dict[key][curr_nodes_idx] for key in layers_dict.keys()} # Get the layers of the graph filtered by the nodes
                curr_pos = pos[curr_nodes_idx]                                                      # Get the positions of the nodes in the graph
                curr_d_pos = curr_pos - curr_pos[0]                                                 # Get the relative positions of the nodes in the graph

                # Define the graph
                graph_dict[central_node_name] = geo_Data(
                    # x=patches[curr_nodes_idx],
                    y=curr_layers[layer],
                    edge_index=curr_edge_index,
                    # edge_attr=curr_edge_attribute,
                    pos=curr_pos,
                    d_pos=curr_d_pos,
                    # patch_names=patch_names[neighbors_dict_index[i]],
                    embeddings=embeddings[curr_nodes_idx] if embeddings is not None else None,
                    predictions=predictions[curr_nodes_idx] if predictions is not None else None,
                    used_mean=used_mean if used_mean is not None else None,
                    num_nodes=len(curr_nodes_idx),
                    mask=layers_dict['mask'][curr_nodes_idx]
                    # **curr_layers
                )

                max_curr_d_pos=curr_d_pos.abs().max()
                if max_curr_d_pos>max_abs_d_pos:
                    max_abs_d_pos=max_curr_d_pos

            #cast as int
            max_abs_d_pos=int(max_abs_d_pos)
            
            # Return the graph dict
            return graph_dict, max_abs_d_pos

        def get_sin_cos_positional_embeddings(self, graph_dict: dict, max_d_pos: int) -> dict:
            
            """This function adds the positional embeddings of each node to the graph dict.

            Args:
                graph_dict (dict): A dictionary where the patch names are the keys and pytorch geometric graph for each one as values
                max_d_pos (int): Max absolute value in the relative position matrix.

            Returns:
                dict: The input graph dict with the information of positional encodings for each graph.
            """
            graph_dict_keys = list(graph_dict.keys())
            embedding_dim =graph_dict[graph_dict_keys[0]].embeddings.shape[1]

            # Define the positional encoding model
            p_encoding_model= PositionalEncoding2D(embedding_dim)

            # Define the empty grid with size (batch_size, x, y, channels)
            grid_size = torch.zeros([1, 2*max_d_pos+1, 2*max_d_pos+1, embedding_dim])

            # Obtain the embeddings for each position
            positional_look_up_table = p_encoding_model(grid_size)        

            for key, value in graph_dict.items():
                d_pos = value.d_pos
                grid_pos = d_pos + max_d_pos
                graph_dict[key].positional_embeddings = positional_look_up_table[0,grid_pos[:,0],grid_pos[:,1],:]
            
            return graph_dict

        print('Computing graphs...')

        # Get unique slide ids
        unique_ids = self.adata.obs['slide_id'].unique()

        # Global dictionary to store the graphs (pytorch geometric graphs)
        graph_dict = {}
        max_global_d_pos=-1

        # Iterate through slides
        for slide in tqdm(unique_ids, leave=True, position=0):
            curr_adata = self.get_slide_from_collection(self.adata, slide)
            curr_graph_dict, max_curr_d_pos = get_graphs_one_slide(self, curr_adata, n_hops, layer, self.hex_geometry)
            
            # Join the current dictionary to the global dictionary
            graph_dict = {**graph_dict, **curr_graph_dict}

            if max_curr_d_pos>max_global_d_pos:
                max_global_d_pos=max_curr_d_pos
        
        graph_dict = get_sin_cos_positional_embeddings(self, graph_dict, max_global_d_pos)

        # Return the graph dict
        return graph_dict
    
    def get_pretrain_dataloaders(self, layer: str = 'd_deltas', batch_size: int = 128, shuffle: bool = True, use_cuda: bool = False) -> Tuple[AnnLoader, AnnLoader, AnnLoader]:
        """
        This function returns the dataloaders for the pre-training phase. This means training a purely vision-based model on only
        the patches to predict the gene expression of the patches.

        Args:
            layer (str, optional): The layer to use for the pre-training. The self.adata.X will be set to that of 'layer'. Defaults to 'deltas'.
            batch_size (int, optional): The batch size of the loaders. Defaults to 128.
            shuffle (bool, optional): Whether to shuffle the data in the loaders. Defaults to True.
            use_cuda (bool, optional): True for using cuda in the loader. Defaults to False.

        Returns:
            Tuple[AnnLoader, AnnLoader, AnnLoader]: The train, validation and test dataloaders. If there is no test set, the test dataloader is None.
        """
        # Get the sample indexes for the train, validation and test sets
        idx_train, idx_val, idx_test = self.adata.obs[self.adata.obs.split == 'train'].index, self.adata.obs[self.adata.obs.split == 'val'].index, self.adata.obs[self.adata.obs.split == 'test'].index

        # Set the X of the adata to the layer caster to float32
        self.adata.X = self.adata.layers[layer].astype(np.float32)

        # Add a binary mask layer specifying valid observations for metric computation
        if 'mask' not in self.adata.layers.keys():
            self.adata.layers['mask'] = self.adata.layers['tpm'] != 0
        # Print with the percentage of the dataset that was replaced
        print('Percentage of imputed observations with median filter: {:5.3f}%'.format(100 * (~self.adata.layers["mask"]).sum() / (self.adata.n_vars*self.adata.n_obs)))

        # If the prediction layer is some form of deltas, add the used mean of the layer as a column in the var
        if 'deltas' in layer:
            # Add a var column of used means of the layer
            mean_key = f'{layer}_avg_exp'.replace('deltas', 'log1p')
            self.adata.var['used_mean'] = self.adata.var[mean_key]

        # Subset the global data handle also the possibility that there is no test set
        adata_train, adata_val = self.adata[idx_train, :], self.adata[idx_val, :]
        adata_test = self.adata[idx_test, :] if len(idx_test) > 0 else None

        # Declare dataloaders
        train_dataloader = AnnLoader(adata_train, batch_size=batch_size, shuffle=shuffle, use_cuda=use_cuda)
        val_dataloader = AnnLoader(adata_val, batch_size=batch_size, shuffle=shuffle, use_cuda=use_cuda)
        test_dataloader = AnnLoader(adata_test, batch_size=batch_size, shuffle=shuffle, use_cuda=use_cuda) if adata_test is not None else None

        return train_dataloader, val_dataloader, test_dataloader
    
    # TODO: Make this function more elegant
    def get_graph_dataloaders(self, layer: str = 'd_deltas', n_hops: int = 2, backbone: str ='densenet', model_path: str = "best_stnet.pt", batch_size: int = 128, shuffle: bool = True) -> Tuple[geo_DataLoader, geo_DataLoader, geo_DataLoader]:
        
        # Get dictionary of parameters to get the graphs
        curr_graph_params = {
            'n_hops': n_hops,
            'layer': layer,
            'backbone': backbone,
            'model_path': model_path
        }        

        # Create graph directory if it does not exist
        os.makedirs(os.path.join(self.dataset_path, 'graphs'), exist_ok=True)
        # Get the filenames of the parameters of all directories in the graph folder
        filenames = glob.glob(os.path.join(self.dataset_path, 'graphs', '**', 'graph_params.json' ), recursive=True)

        # Define boolean to check if the graphs are already saved
        found_graphs = False

        # Iterate over all the filenames and check if the parameters are the same
        for filename in filenames:
            with open(filename, 'r') as f:
                # Load the parameters of the dataset
                saved_params = json.load(f)
                # Check if the parameters are the same
                if saved_params == curr_graph_params:
                    print(f'Graph data already saved in {filename}')
                    found_graphs = True
                    # Track the time and load the graphs
                    start = time()
                    train_graphs = torch.load(os.path.join(os.path.dirname(filename), 'train_graphs.pt'))
                    val_graphs = torch.load(os.path.join(os.path.dirname(filename), 'val_graphs.pt'))
                    test_graphs = torch.load(os.path.join(os.path.dirname(filename), 'test_graphs.pt')) if os.path.exists(os.path.join(os.path.dirname(filename), 'test_graphs.pt')) else None
                    print(f'Loaded graphs in {time() - start:.2f} seconds.')
                    break

        # If the graphs are not found, compute them
        if not found_graphs:
            
            # Print that we are computing the graphs
            print('Graphs not found in file, computing graphs...')

            # We compute the embeddings and predictions for the patches
            self.compute_patches_embeddings_and_predictions(preds=True, backbone=backbone, model_path=model_path)
            self.compute_patches_embeddings_and_predictions(preds=False, backbone=backbone, model_path=model_path)
            
            # Get graph dicts
            general_graph_dict = self.get_graphs(n_hops=n_hops, layer=layer)

            # Get the train, validation and test indexes
            idx_train, idx_val, idx_test = self.adata.obs[self.adata.obs.split == 'train'].index, self.adata.obs[self.adata.obs.split == 'val'].index, self.adata.obs[self.adata.obs.split == 'test'].index

            # Get list of graphs
            train_graphs = [general_graph_dict[idx] for idx in idx_train]
            val_graphs = [general_graph_dict[idx] for idx in idx_val]
            test_graphs = [general_graph_dict[idx] for idx in idx_test] if len(idx_test) > 0 else None

            print('Saving graphs...')
            # Create graph directory if it does not exist with the current time
            graph_dir = os.path.join(self.dataset_path, 'graphs', datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
            os.makedirs(graph_dir, exist_ok=True)

            # Save the graph parameters
            with open(os.path.join(graph_dir, 'graph_params.json'), 'w') as f:
                # Write the json
                json.dump(curr_graph_params, f, indent=4)

            torch.save(train_graphs, os.path.join(graph_dir, 'train_graphs.pt'))
            torch.save(val_graphs, os.path.join(graph_dir, 'val_graphs.pt'))
            torch.save(test_graphs, os.path.join(graph_dir, 'test_graphs.pt')) if test_graphs is not None else None
        

        # Declare dataloaders
        train_dataloader = geo_DataLoader(train_graphs, batch_size=batch_size, shuffle=shuffle)
        val_dataloader = geo_DataLoader(val_graphs, batch_size=batch_size, shuffle=shuffle)
        test_dataloader = geo_DataLoader(test_graphs, batch_size=batch_size, shuffle=shuffle) if test_graphs is not None else None

        return train_dataloader, val_dataloader, test_dataloader
        

class HisToGeneDataset(Dataset):
    def __init__(self, adata, set_str):
        self.set = set_str
        self.adata = adata[adata.obs.split == self.set]
        self.idx_2_slide = {idx: slide for idx, slide in enumerate(self.adata.obs.slide_id.unique())}
        
        #Perform transformations
        self.transforms = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        tissue_tiles = self.adata.obsm['patches_scale_1.0']
        # Pass to torch tensor
        tissue_tiles = torch.from_numpy(tissue_tiles)
        w = round(np.sqrt(tissue_tiles.shape[1]/3))
        tissue_tiles = tissue_tiles.reshape((tissue_tiles.shape[0], w, w, -1))
        # Permute dimensions to be in correct order for normalization
        tissue_tiles = tissue_tiles.permute(0,3,1,2).contiguous()
        # Make transformations in tissue tiles
        tissue_tiles = tissue_tiles/255.
        # Transform tiles
        tissue_tiles = self.transforms(tissue_tiles)
        # Flatten tiles
        self.adata.obsm['patches_scale_1.0_transformed'] = tissue_tiles.view(tissue_tiles.shape[0], -1)
        self.adata.obsm['patches_scale_1.0_transformed_numpy'] = tissue_tiles.view(tissue_tiles.shape[0], -1).numpy()

        # Define mask layer
        self.adata.layers['mask'] = self.adata.layers['tpm'] != 0

    def __len__(self):
        return len(self.idx_2_slide)

    def __getitem__(self, idx):
        
        # Get the slide from the index
        slide = self.idx_2_slide[idx]
        # Get the adata of the slide
        adata_slide = self.adata[self.adata.obs.slide_id == slide]

        # Get the patches
        patch = torch.from_numpy(adata_slide.obsm['patches_scale_1.0_transformed_numpy'])
        # Get the coordinates
        coor = torch.from_numpy(adata_slide.obs[['array_row', 'array_col']].values)
        # Get the expression
        exp = torch.from_numpy(adata_slide.X.toarray())
        # Get the mask
        mask = torch.from_numpy(adata_slide.layers['mask'])
        
        return patch, coor, exp, mask


# Test code only for debugging
if __name__ == "__main__":
    
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

    test_dataset = SpatialDataset(
        dataset =       config['dataset'], 
        param_dict =    param_dict, 
        patch_scale =   config['patch_scale'], 
        patch_size =    config['patch_size'], 
        force_compute = config['force_compute']
    )
    breakpoint()
    # Get the dataloaders
    train_dataloader, val_dataloader, test_dataloader = test_dataset.get_pretrain_dataloaders()
    # train_dataloader, val_dataloader, test_dataloader = test_dataset.get_graph_dataloaders(layer='c_d_deltas', backbone="ViT", model_path=os.path.join("pretrained_ie_models","best_deltas_stnet.pt"), n_hops=1, batch_size=128, shuffle=True)




    
