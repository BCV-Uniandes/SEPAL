import numpy as np
import torch
import warnings
from typing import Union
import warnings
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from time import time

warnings.filterwarnings(action='ignore', category=UserWarning)

def pearsonr_cols(gt_mat: torch.Tensor, pred_mat: torch.Tensor, mask: torch.Tensor) -> float:
    """
    This function receives 2 matrices of shapes (n_observations, n_variables) and computes the average Pearson correlation.
    To do that, it takes the i-th column of each matrix and computes the Pearson correlation between them.
    It finally returns the average of all the Pearson correlations computed.

    Args:
        gt_mat (torch.Tensor): Ground truth matrix of shape (n_observations, n_variables).
        pred_mat (torch.Tensor): Predicted matrix of shape (n_observations, n_variables).
        mask (torch.Tensor): Boolean mask with False in positions that must be ignored in metric computation (n_observations, n_variables).
    
    Returns:
        mean_pcc (float): Mean Pearson correlation computed by averaging the Pearson correlation for each patch.
    """
    masked_gt_mat = torch.masked.masked_tensor(gt_mat, mask=mask)
    masked_gt_mean = masked_gt_mat.mean(dim=0, keepdim=True)

    masked_pred_mat = torch.masked.masked_tensor(pred_mat, mask=mask)
    masked_pred_mean = masked_pred_mat.mean(dim=0, keepdim=True)

    # Construct matrices with only masked means
    masked_gt_mean = masked_gt_mean.to_tensor(float('nan')).repeat(gt_mat.shape[0],1)
    masked_pred_mean = masked_pred_mean.to_tensor(float('nan')).repeat(pred_mat.shape[0],1)

    # Modify mask==False entries of gt_mat and pred_mat to the masked mean. 
    # NOTE: This replace will make the computation of the metric efficient without taking into account the discarded values of the mask
    gt_mat = torch.where(mask==True, gt_mat, masked_gt_mean)
    pred_mat = torch.where(mask==True, pred_mat, masked_pred_mean)

    # Center both matrices by subtracting the mean of each column
    centered_gt_mat = gt_mat - masked_gt_mean
    centered_pred_mat = pred_mat - masked_pred_mean

    # Compute pearson correlation with cosine similarity
    pcc = torch.nn.functional.cosine_similarity(centered_gt_mat, centered_pred_mat, dim=0)

    # Compute mean pearson correlation
    mean_pcc = pcc.mean().item()

    return mean_pcc

def pearsonr_gene(gt_mat: torch.Tensor, pred_mat: torch.Tensor, mask: torch.Tensor) -> float:
    """
    This function uses pearsonr_cols to compute the Pearson correlation between the ground truth and predicted matrices along
    the gene dimension. It is computing the correlation between the true and predicted values for each gene and returning the average of all.

    Args:
        gt_mat (torch.Tensor): Ground truth matrix of shape (n_samples, n_genes).
        pred_mat (torch.Tensor): Predicted matrix of shape (n_samples, n_genes).
        mask (torch.Tensor): Boolean mask with False in positions that must be ignored in metric computation (n_samples, n_genes).

    Returns:
        float: Mean Pearson correlation computed by averaging the Pearson correlation for each gene.
    """
    return pearsonr_cols(gt_mat=gt_mat, pred_mat=pred_mat, mask=mask)

def pearsonr_patch(gt_mat: torch.Tensor, pred_mat: torch.Tensor, mask: torch.Tensor) -> float:
    """
    This function uses pearsonr_cols to compute the Pearson correlation between the ground truth and predicted matrices along
    the patch dimension. It is computing the correlation the between true and predicted values for each patch and returning the average of all.

    Args:
        gt_mat (torch.Tensor): Ground truth matrix of shape (n_samples, n_genes).
        pred_mat (torch.Tensor): Predicted matrix of shape (n_samples, n_genes).
        mask (torch.Tensor): Boolean mask with False in positions that must be ignored in metric computation (n_samples, n_genes).

    Returns:
        float: Mean Pearson correlation computed by averaging the Pearson correlation for each patch.
    """
    # Transpose matrices and apply pearsonr_torch_cols 
    return pearsonr_cols(gt_mat=gt_mat.T, pred_mat=pred_mat.T, mask=mask.T)

def r2_score_cols(gt_mat: torch.Tensor, pred_mat: torch.Tensor, mask: torch.Tensor) -> float:
    """
    This function receives 2 matrices of shapes (n_observations, n_variables) and computes the average R2 score.
    To do that, it takes the i-th column of each matrix and computes the R2 score between them.
    It finally returns the average of all the R2 scores computed.

    Args:
        gt_mat (torch.Tensor): Ground truth matrix of shape (n_observations, n_variables).
        pred_mat (torch.Tensor): Predicted matrix of shape (n_observations, n_variables).
        mask (torch.Tensor): Boolean mask with False in positions that must be ignored in metric computation (n_observations, n_variables).

    Returns:
        float: Mean R2 score computed by averaging the R2 score for each column in the matrices.
    """

    # Pass input matrices to masked tensors
    gt_mat = torch.masked.masked_tensor(gt_mat, mask=mask)
    pred_mat = torch.masked.masked_tensor(pred_mat, mask=mask)

    # Compute the column means of the ground truth
    gt_col_means = gt_mat.mean(dim=0).to_tensor(float('nan'))
    
    # Compute the total sum of squares
    total_sum_squares = torch.sum(torch.square(gt_mat - gt_col_means), dim=0).to_tensor(float('nan'))

    # Compute the residual sum of squares
    residual_sum_squares = torch.sum(torch.square(gt_mat - pred_mat), dim=0).to_tensor(float('nan'))

    # Compute the R2 score for each column
    r2_scores = 1. - (residual_sum_squares / total_sum_squares)

    # Compute the mean R2 score
    mean_r2_score = r2_scores.mean().item()

    return mean_r2_score

def r2_score_gene(gt_mat: torch.Tensor, pred_mat: torch.Tensor, mask: torch.Tensor) -> float:
    """
    This function uses r2_score_cols to compute the R2 score between the ground truth and predicted matrices along
    the gene dimension. It is computing the R2 score between the true and predicted values for each gene and returning the average of all.

    Args:
        gt_mat (torch.Tensor): Ground truth matrix of shape (n_samples, n_genes).
        pred_mat (torch.Tensor): Predicted matrix of shape (n_samples, n_genes).
        mask (torch.Tensor): Boolean mask with False in positions that must be ignored in metric computation (n_samples, n_genes).

    Returns:
        float: Mean R2 score computed by averaging the R2 score for each gene.
    """
    return r2_score_cols(gt_mat=gt_mat, pred_mat=pred_mat, mask=mask)

def r2_score_patch(gt_mat: torch.Tensor, pred_mat: torch.Tensor, mask: torch.Tensor) -> float:
    """
    This function uses r2_score_cols to compute the R2 score between the ground truth and predicted matrices along
    the patch dimension. It is computing the R2 score between the true and predicted values for each patch and returning the average of all.

    Args:
        gt_mat (torch.Tensor): Ground truth matrix of shape (n_samples, n_genes).
        pred_mat (torch.Tensor): Predicted matrix of shape (n_samples, n_genes).
        mask (torch.Tensor): Boolean mask with False in positions that must be ignored in metric computation (n_samples, n_genes).

    Returns:
        float: Mean R2 score computed by averaging the R2 score for each patch.
    """
    # Transpose matrices and apply r2_score_torch_cols
    return r2_score_cols(gt_mat=gt_mat.T, pred_mat=pred_mat.T, mask=mask.T)


def slow_pearsonr_cols(gt_mat: torch.Tensor, pred_mat: torch.Tensor, mask: torch.Tensor) -> float:
    """
    This is a function that makes exactly the same as pearsonr_cols but with scipy and not using tricks for speed up.
    It is used to check that the results are the same.

    Args:
        gt_mat (torch.Tensor): Ground truth matrix of shape (n_observations, n_variables).
        pred_mat (torch.Tensor): Predicted matrix of shape (n_observations, n_variables).
        mask (torch.Tensor): Boolean mask with False in positions that must be ignored in metric computation (n_observations, n_variables).
    
    Returns:
        mean_pcc (float): Mean Pearson correlation computed by averaging the Pearson correlation for each patch.
    """

    pcc_list = []

    for i in range(gt_mat.shape[1]):
        curr_mask = mask[:, i]

        gt_col = gt_mat[curr_mask, i].numpy()
        pred_col = pred_mat[curr_mask, i].numpy()

        pcc = pearsonr(gt_col, pred_col)[0]
        pcc_list.append(pcc)

    mean_pcc = np.mean(pcc_list)

    return mean_pcc

def slow_pearsonr_gene(gt_mat: torch.Tensor, pred_mat: torch.Tensor, mask: torch.Tensor) -> float:
    """
    This is a slower twin of the function pearsonr_gene. It is used to check that the results are the same.
    """
    return slow_pearsonr_cols(gt_mat=gt_mat, pred_mat=pred_mat, mask=mask)

def slow_pearsonr_patch(gt_mat: torch.Tensor, pred_mat: torch.Tensor, mask: torch.Tensor) -> float:
    """
    This is a slower twin of the function pearsonr_patch. It is used to check that the results are the same.
    """
    return slow_pearsonr_cols(gt_mat=gt_mat.T, pred_mat=pred_mat.T, mask=mask.T)

def slow_r2_score_cols(gt_mat: torch.Tensor, pred_mat: torch.Tensor, mask: torch.Tensor) -> float:
    """
    This is a function that makes exactly the same as r2_score_cols but with sklearn and not using tricks for speed up.
    It is used to check that the results are the same.

    Args:
        gt_mat (torch.Tensor): Ground truth matrix of shape (n_observations, n_variables).
        pred_mat (torch.Tensor): Predicted matrix of shape (n_observations, n_variables).
        mask (torch.Tensor): Boolean mask with False in positions that must be ignored in metric computation (n_observations, n_variables).

    Returns:
        float: Mean R2 score computed by averaging the R2 score for each column in the matrices.
    """
    r2_scores = []

    for i in range(gt_mat.shape[1]):
        curr_mask = mask[:, i]

        gt_col = gt_mat[curr_mask, i].numpy()
        pred_col = pred_mat[curr_mask, i].numpy()

        curr_r2_score = r2_score(gt_col, pred_col)
        r2_scores.append(curr_r2_score)

    mean_r2_score = np.mean(r2_scores)

    return mean_r2_score

def slow_r2_score_gene(gt_mat: torch.Tensor, pred_mat: torch.Tensor, mask: torch.Tensor) -> float:
    """
    This is a slower twin of the function r2_score_gene. It is used to check that the results are the same.
    """
    return slow_r2_score_cols(gt_mat=gt_mat, pred_mat=pred_mat, mask=mask)

def slow_r2_score_patch(gt_mat: torch.Tensor, pred_mat: torch.Tensor, mask: torch.Tensor) -> float:
    """
    This is a slower twin of the function r2_score_patch. It is used to check that the results are the same.
    """
    return slow_r2_score_cols(gt_mat=gt_mat.T, pred_mat=pred_mat.T, mask=mask.T)


def get_metrics(gt_mat: Union[np.array, torch.Tensor] , pred_mat: Union[np.array, torch.Tensor], mask: Union[np.array, torch.Tensor]) -> dict:
    """
    This function receives 2 matrices of shapes (n_samples, n_genes) and computes the following metrics:
    
        - Pearson correlation (gene-wise) [PCC-Gene]
        - Pearson correlation (patch-wise) [PCC-Patch]
        - r2 score (gene-wise) [R2-Gene]
        - r2 score (patch-wise) [R2-Patch]
        - Mean squared error [MSE]
        - Mean absolute error [MAE]
        - Global metric [Global] (Global = PCC-Gene + R2-Gene + PCC-Patch + R2-Patch - MAE - MSE)

    Args:
        gt_mat (Union[np.array, torch.Tensor]): Ground truth matrix of shape (n_samples, n_genes).
        pred_mat (Union[np.array, torch.Tensor]): Predicted matrix of shape (n_samples, n_genes).
        mask (Union[np.array, torch.Tensor]): Boolean mask with False in positions that must be ignored in metric computation (n_samples, n_genes).

    Returns:
        dict: Dictionary containing the metrics computed. The keys are: ['PCC-Gene', 'PCC-Patch', 'R2-Gene', 'R2-Patch', 'MSE', 'MAE', 'Global']
    """

    # Assert that all matrices have the same shape
    assert gt_mat.shape == pred_mat.shape, "gt_mat and pred_mat matrices must have the same shape."
    assert gt_mat.shape == mask.shape, "gt_mat and mask matrices must have the same shape."

    # If input are numpy arrays, convert them to torch tensors
    if isinstance(gt_mat, np.ndarray):
        gt_mat = torch.from_numpy(gt_mat)
    if isinstance(pred_mat, np.ndarray):
        pred_mat = torch.from_numpy(pred_mat)
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    # Get boolean indicating constant columns in predicted matrix 
    # NOTE: A constant gene prediction will mess with the pearson correlation
    constant_cols = torch.all(pred_mat == pred_mat[[0],:], axis = 0)
    # Get boolean indicating if there are any constant columns
    any_constant_cols = torch.any(constant_cols)

    # Get boolean indicating constant rows in predicted matrix
    # NOTE: A constant patch prediction will mess with the pearson correlation
    constant_rows = torch.all(pred_mat == pred_mat[:,[0]], axis = 1)
    # Get boolean indicating if there are any constant rows
    any_constant_rows = torch.any(constant_rows)

    # If there are any constant columns, set the pcc_g and r2_g to None
    if any_constant_cols:
        pcc_g = None
        warnings.warn("There are constant columns in the predicted matrix. This means a gene is being predicted as constant. The Pearson correlation (gene-wise) will be set to None.")
    else:
        # Compute Pearson correlation (gene-wise)
        pcc_g = pearsonr_gene(gt_mat, pred_mat, mask=mask)
    
    # If there are any constant rows, set the pcc_p and r2_p to None
    if any_constant_rows:
        pcc_p = None
        warnings.warn("There are constant rows in the predicted matrix. This means a patch is being predicted as constant. The Pearson correlation (patch-wise) will be set to None.")
    else:
        # Compute Pearson correlation (patch-wise)
        pcc_p = pearsonr_patch(gt_mat, pred_mat, mask=mask)
        

    # Compute r2 score (gene-wise)
    r2_g = r2_score_gene(gt_mat, pred_mat, mask=mask)
    # Compute r2 score (patch-wise)
    r2_p = r2_score_patch(gt_mat, pred_mat, mask=mask)

    # Compute mean squared error
    mse = torch.nn.functional.mse_loss(gt_mat[mask], pred_mat[mask], reduction='mean').item()
    # Compute mean absolute error
    mae = torch.nn.functional.l1_loss(gt_mat[mask], pred_mat[mask], reduction='mean').item()

    # Create dictionary with the metrics computed
    metrics_dict = {
        'PCC-Gene': pcc_g,
        'PCC-Patch': pcc_p,
        'R2-Gene': r2_g,
        'R2-Patch': r2_p,
        'MSE': mse,
        'MAE': mae,
        'Global': pcc_g + pcc_p + r2_g + r2_p - mse - mae 
    }

    return metrics_dict

def slow_get_metrics(gt_mat: Union[np.array, torch.Tensor] , pred_mat: Union[np.array, torch.Tensor], mask: Union[np.array, torch.Tensor]) -> dict:
    """
    This is a slower twin of the function get_metrics. It is used to check that the results are the same.
    """

    # Assert that all matrices have the same shape
    assert gt_mat.shape == pred_mat.shape, "gt_mat and pred_mat matrices must have the same shape."
    assert gt_mat.shape == mask.shape, "gt_mat and mask matrices must have the same shape."

    # If input are numpy arrays, convert them to torch tensors
    if isinstance(gt_mat, np.ndarray):
        gt_mat = torch.from_numpy(gt_mat)
    if isinstance(pred_mat, np.ndarray):
        pred_mat = torch.from_numpy(pred_mat)
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    # Get boolean indicating constant columns in predicted matrix 
    # NOTE: A constant gene prediction will mess with the pearson correlation
    constant_cols = torch.all(pred_mat == pred_mat[[0],:], axis = 0)
    # Get boolean indicating if there are any constant columns
    any_constant_cols = torch.any(constant_cols)

    # Get boolean indicating constant rows in predicted matrix
    # NOTE: A constant patch prediction will mess with the pearson correlation
    constant_rows = torch.all(pred_mat == pred_mat[:,[0]], axis = 1)
    # Get boolean indicating if there are any constant rows
    any_constant_rows = torch.any(constant_rows)

    # If there are any constant columns, set the pcc_g and r2_g to None
    if any_constant_cols:
        pcc_g = None
        warnings.warn("There are constant columns in the predicted matrix. This means a gene is being predicted as constant. The Pearson correlation (gene-wise) will be set to None.")
    else:
        # Compute Pearson correlation (gene-wise)
        pcc_g = slow_pearsonr_gene(gt_mat, pred_mat, mask=mask)
    
    # If there are any constant rows, set the pcc_p and r2_p to None
    if any_constant_rows:
        pcc_p = None
        warnings.warn("There are constant rows in the predicted matrix. This means a patch is being predicted as constant. The Pearson correlation (patch-wise) will be set to None.")
    else:
        # Compute Pearson correlation (patch-wise)
        pcc_p = slow_pearsonr_patch(gt_mat, pred_mat, mask=mask)
        

    # Compute r2 score (gene-wise)
    r2_g = slow_r2_score_gene(gt_mat, pred_mat, mask=mask)
    # Compute r2 score (patch-wise)
    r2_p = slow_r2_score_patch(gt_mat, pred_mat, mask=mask)

    # Compute mean squared error
    mse = torch.nn.functional.mse_loss(gt_mat[mask], pred_mat[mask], reduction='mean').item()
    # Compute mean absolute error
    mae = torch.nn.functional.l1_loss(gt_mat[mask], pred_mat[mask], reduction='mean').item()

    # Create dictionary with the metrics computed
    metrics_dict = {
        'PCC-Gene': pcc_g,
        'PCC-Patch': pcc_p,
        'R2-Gene': r2_g,
        'R2-Patch': r2_p,
        'MSE': mse,
        'MAE': mae,
        'Global': pcc_g + pcc_p + r2_g + r2_p - mse - mae 
    }

    return metrics_dict


# Here we have some testing code
if __name__=='__main__':
    
    # Set number of observations and genes (hypothetical)
    obs = 7777
    genes = 256
    imputed_fraction = 0.26 # This is the percentage of zeros in the mask

    # Henerate random matrices
    pred = torch.randn((obs,genes))
    gt = torch.randn((obs,genes))
    mask = torch.rand((obs,genes))>imputed_fraction

    # Compute metrics with the fast way (efficient implementation)
    print('Fast metrics'+'-'*40)
    start = time()
    test_metrics = get_metrics(gt, pred, mask=mask)
    print("Time taken: {:5.2f}s".format(time()-start))

    for key, val in test_metrics.items():
        print("{} = {:5.7f}".format(key, val))


    # Compute metrics with the slow way (inefficient implementation but secure)
    print('Slow metrics'+'-'*40) 
    start = time()
    slow_test_metrics = slow_get_metrics(gt, pred, mask=mask)
    print("Time taken: {:5.2f}s".format(time()-start))

    for key, val in slow_test_metrics.items():
        print("{} = {:5.7f}".format(key, val))
