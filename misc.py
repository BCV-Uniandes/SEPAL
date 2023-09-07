import anndata as ad
import numpy as np
from combat.pycombat import pycombat
import pandas as pd
import torch

# compute = 'none' # 1, 2, none

# if (compute == '1') or (compute == '2'):
#     # Set numpy seed
#     np.random.seed(0)
#     # Get 100 by 100 matrix
#     mat = np.random.rand(256, 29820)
#     # Get 100 by 1 vector of zeros and ones
#     vec = np.random.randint(0, 23, 29820)

#     corrected = pycombat(pd.DataFrame(15+mat), vec, par_prior=True)

#     save_path = "corrected_1.csv" if compute == '1' else "corrected_2.csv"
#     corrected.to_csv(save_path, index=False)

# else:
#     corrected_1 = pd.read_csv("corrected_1.csv")
#     corrected_2 = pd.read_csv("corrected_2.csv")
#     print(f"corrected_1 max diff corrected_2: {(corrected_1-corrected_2).mean().mean()}")

# breakpoint()

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


# Define path to datasets
ref_path = "/media/SSD3/gmmejia/ST/stability_check/adata_stnet_lambda004_1.h5ad"
quety_path = "/media/SSD3/gmmejia/ST/stability_check/adata_stnet_lambda003_1.h5ad"

# Load datasets
ref = ad.read_h5ad(ref_path)
query = ad.read_h5ad(quety_path)


print(f"counts max diff query to ref: {np.max(np.abs(query.layers['counts']-ref.layers['counts']))}")
print(f"tpm max diff query to ref: {np.max(np.abs(query.layers['tpm']-ref.layers['tpm']))}")
print(f"log1p max diff query to ref: {np.max(np.abs(query.layers['log1p']-ref.layers['log1p']))}")
print(f"d_log1p max diff query to ref: {np.max(np.abs(query.layers['d_log1p']-ref.layers['d_log1p']))}")
print(f"c_log1p max diff query to ref: {np.max(np.abs(query.layers['c_log1p']-ref.layers['c_log1p']))}")
print(f"c_d_log1p max diff query to ref: {np.max(np.abs(query.layers['c_d_log1p']-ref.layers['c_d_log1p']))}")




# Redo combat transformation
# ref = combat_transformation(ref, "patient", "d_log1p", "test_combat")
# curr = combat_transformation(curr, "patient", "d_log1p", "test_combat")

# print(f"test_combat max diff ref to curr: {np.max(np.abs(ref.layers['test_combat']-curr.layers['test_combat']))}")
