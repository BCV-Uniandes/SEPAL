# Pytorch implementation of SEPAL: Spatial Gene Expression Prediction from Local Graphs

### Authored by Gabriel Mejia, Paula Cardenas*, Daniela Ruiz*, Angela Castillo and Pablo Arbeláez

![](Figures/OVERVIEW.png)

**Abstract**

Spatial transcriptomics is an emerging technology that aligns histopathology images with spatially resolved gene expression profiling. It holds the potential for understanding many diseases but faces significant bottlenecks such as specialized equipment and domain expertise. In this work, we present SEPAL, a new model for predicting genetic profiles from visual tissue appearance. Our method exploits the biological biases of the problem by directly supervising relative differences with respect to mean expression, and leverages local visual context at every coordinate to make predictions using a graph neural network. This approach closes the gap between complete locality and complete globality in current methods. In addition, we propose a novel benchmark that aims to better define the task by following current best practices in transcriptomics and restricting the prediction variables to only those with clear spatial patterns. Our extensive evaluation in two different human breast cancer datasets indicates that SEPAL outperforms previous state-of-the-art methods and other mechanisms of including spatial context.

## News:

* _8/7/2023_: SEPAL has been accepted as an **oral presentation** in the ICCV workshop of [Computer Vision for Automated Medical Diagnosis](https://cvamd2023.github.io/)
* _9/7/2023_: SEPAL preprint is available [here](https://doi.org/10.48550/arXiv.2309.01036)

## Set up

Run the following to define your environment in the terminal:

```bash
conda create -n st
conda activate st
conda install python==3.10.0
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install squidpy
pip install wandb
pip install wget
pip install combat
pip install opencv-python
pip install --upgrade tbb
pip install positional-encodings[pytorch]
```

The datasets and gene annotation files are downloaded automatically.

We use the GTF file of the [basic gene annotation](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/gencode.v43.basic.annotation.gtf.gz) from [GENCODE](https://www.gencodegenes.org/human/) and use the set of functions provided by [GTFtools](https://www.genemine.org/gtftools.php) to compute the genes length in TPM normalization. 


## Main Results

To obtain the main results you must first train the image encoder using the following line:

```bash
python run_main_config.py --dataset_config configs/datasets/$dataset_name$_deltas.json --model_config configs/models/best_vit_$dataset_name$.json --train_config configs/training/best_vit_$dataset_name$.json 
```

Where the substring `$dataset_name$` must be replaced by `visium` or `stnet_dataset` depending on what you want to try. This will train the image encoder and save the results inside the `results/best_vit_$dataset_name$` directory. Now, to train the spatial module you must run:

```bash
python run_main_config.py --dataset_config configs/datasets/$dataset_name$_deltas.json --model_config configs/models/best_sepal_$dataset_name$.json --train_config configs/training/best_sepal_$dataset_name$.json 
```

This will train the graph neural network and save the results inside the `results/best_sepal_$dataset_name$` directory. Your results might differ a little from the values reported in the paper. Consequently, we also provide our pre-trained models in the binary release of the code. The configurations are the same as before `configs/models/best_sepal_$dataset_name$.json` but you must change the paths where the image encoders are loaded accordingly. Retraining is mandatory when using another dataset since the specific predictor genes selected by moran scores may vary.

When trained, the programs will prompt the possibility to log results into a weights and biases (W&B) account. We encourage its use since most of the logging performed inside was designed to interface with W&B.
