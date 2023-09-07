import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn
from torchvision import models      
   

class st_network(torch.nn.Module):
    #TODO: add dropout
    def __init__(self, act, graph_operator, h_preprocess, h_graph, h_pred_head, pooling, sum_positions) -> None:
        """_summary_

        Args:
            act (str): Activation function used in the MLP
            graph_operator (str): Graph convolutional operator (i.e. chebconv)
            h_preprocess (list): List of channels for the preprocessing MLP
            h_graph (list): List of channels for the GNN
            h_pred_head (list): List of channels for the prediction head MLP
            pooling (str): Pooling operator (i.e. global_mean_pool)
        """
        super(st_network,self).__init__()
        
        self.act = act
        self.graph_operator = graph_operator
        self.h_preprocess = h_preprocess
        self.h_graph = h_graph
        self.h_pred_head = h_pred_head
        self.pooling = pooling
        self.sum_positions = sum_positions

        self.act_fn = getattr(nn, act)()

        self.graph_operator_fn = getattr(gnn, self.graph_operator)
        self.pooling_fn = getattr(gnn, self.pooling)(in_channels=self.h_graph[-1], ratio=1) if self.pooling == "SAGPooling" else getattr(gnn, self.pooling)

        self.preprocess_layer = MLP(self.h_preprocess, self.act) if self.h_preprocess is not [-1] else nn.Identity()

        # Convolution definitions
        self.layers = nn.ModuleList()
        for i in range(len(self.h_graph)-1):
            self.layers.append(self.graph_operator_fn(self.h_graph[i], self.h_graph[i+1]))
        
        self.prediction_layer = MLP(self.h_pred_head, self.act) if self.h_pred_head is not [-1] else nn.Identity()

    def forward (self, graph):
    
        emb_matrix = graph.embeddings
        pos_embs_matrix = graph.positional_embeddings

        if self.sum_positions:
            ftr_matrix = torch.add(emb_matrix, pos_embs_matrix)

        else:
            ftr_matrix = torch.cat((emb_matrix, pos_embs_matrix), dim=1)
        
        x = self.preprocess_layer(ftr_matrix)
        
        for layer in self.layers:
            x = layer(x, graph.edge_index)
            x = self.act_fn(x)

        if self.pooling == "SAGPooling":
            x, _, _, _, _, _ = self.pooling_fn(x, edge_index=graph.edge_index, batch=graph.batch) 
        else:
            x = self.pooling_fn(x, batch=graph.batch)

        out = self.prediction_layer(x)

        return out

class baseline_regressor(torch.nn.Module):
    def __init__(self, layer_list, act):
        """
        Baseline model that is the most naive approach to the reconstruction problem. It basically takes single cell RNAseq inputs
        and returns an X, Y coordinate using an MLP.

        Args:
            layer_list (list):  List of ints defining the sizes of the layers in the MLP. The first element must be the input size and the
                                last component must be the output size which is 2. 
            act (str):          String with the activation function to use. Options are all activation functions defined in: 
                                https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
                                examples are: 'ELU', 'ReLU', 'Hardtanh'... the string is case sensitive with respect to the ones
                                defined at PyTorch website.

        """
        super(baseline_regressor, self).__init__()

        # Activation function definition
        self.activation = getattr(nn, act)()

        # Define hidden sizes list
        self.mlp_dims = layer_list

        # Define layers
        # The output layer is the last element in self.layers
        self.layers = nn.ModuleList()
        for i in range(len(self.mlp_dims)-1):
            self.layers.append(nn.Linear(self.mlp_dims[i], self.mlp_dims[i+1]))

   
    # FIXME: right now just expression is entering. Input should be standard between all models (must include patches as input but not use them)
    def forward(self, expression):
        """
        Performs forward pass of the baseline regressor. It just uses the scRNAseq data

        Args:
            expression (tensor): Matrix of shape (batch_size, gene_number)

        Returns:
            tensor: A tensor matrix where dimensions are (batch_size, output_size). Output_size is the last element in self.mlp_dims  
        """

        x = expression

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        out = x

        return out


class MLP(torch.nn.Module):
    def __init__(self, layer_list, act):
        """
        Generic multilayer perceptron where the dimensions of layers can be specified along with the activation function.

        Args:
            layer_list (list):  List of ints defining the sizes of the layers in the MLP. The first element must be the input size and the
                                last component must be the output size.
            act (str):          String with the activation function to use. Options are all activation functions defined in: 
                                https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
                                examples are: 'ELU', 'ReLU', 'Hardtanh'... the string is case sensitive with respect to the ones
                                defined at PyTorch website.
        """
        super(MLP, self).__init__()

        # Activation function definition
        self.activation = getattr(nn, act)()

        # Define hidden sizes list
        self.mlp_dims = layer_list

        # Define layers
        # The output layer is the last element in self.layers
        self.layers = nn.ModuleList()
        for i in range(len(self.mlp_dims)-1):
            self.layers.append(nn.Linear(self.mlp_dims[i], self.mlp_dims[i+1]))
    
    def forward(self, expression):
        """
        Performs forward pass of the MLP.

        Args:
            expression (tensor): Matrix of shape (batch_size, features)

        Returns:
            tensor: A tensor matrix where dimensions are (batch_size, output_size). Output_size is the last element in self.mlp_dims  
        """

        x = expression

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        out = x

        return out


class autoEncoder(torch.nn.Module):
    def __init__(self, layer_list, act):
        """
        This is a generic auto-encoder where you can specify the layers dimensions. This module is symmetric. Hence,
        the decoder will have the same layer dimensions as the encoder but in reverse order. The layer list just has the
        dimensions of the encoder. The last number of the layer_list is the dimension of the latent space.

        Args:
            layer_list (list): List of ints defining the sizes of the layers in the MLP encoder. The first element must be the input size and the
                                last component must be the size of the latent space. The decoder will have the same dimensions but
                                reversed.
            act (str):          String with the activation function to use. Options are all activation functions defined in: 
                                https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
                                examples are: 'ELU', 'ReLU', 'Hardtanh'... the string is case sensitive with respect to the ones
                                defined at PyTorch website.
        """
        super(autoEncoder, self).__init__()

        self.act = act
        self.encoder_dims = layer_list
        self.decoder_dims = layer_list[::-1]

        self.encoder = MLP(self.encoder_dims, self.act)
        self.decoder = MLP(self.decoder_dims, self.act)

    def forward(self, expression):
        """
        Performs a forward pass of the auto encoder and return both the reconstruction and the latent space. 

        Args:
            expression (torch.Tensor): Matrix of shape (batch_size, features)

        Returns:
            reconstruction (torch.Tensor): The final reconstruction of expression after passing for the bottleneck. Of shape (batch_size, features).
            latent_space (torch.Tensor): The intermediate representation of the data in the latent space after the encoder. Or shape (batch_size, self.encoder_dims[-1])
        """
        latent_space = self.encoder(expression)
        reconstruction = self.decoder(latent_space)

        return reconstruction, latent_space


class ImageEncoder(torch.nn.Module):
    def __init__(self, backbone, use_pretrained,  latent_dim):

        super(ImageEncoder, self).__init__()

        self.backbone = backbone
        self.use_pretrained = use_pretrained
        self.latent_dim = latent_dim

        # Initialize the model using various options 
        self.encoder, self.input_size = self.initialize_model()

    def initialize_model(self):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        model_weights = 'IMAGENET1K_V1' if self.use_pretrained else None
        input_size = 0

        if self.backbone == "resnet": ##
            """ Resnet18 acc@1 (on ImageNet-1K): 69.758
            """
            model_ft = models.resnet18(weights=model_weights)   #Get model
            num_ftrs = model_ft.fc.in_features                  #Get in features of the fc layer (final layer)
            model_ft.fc = nn.Linear(num_ftrs, self.latent_dim)  #Keep in features, but modify out features for self.latent_dim
            input_size = 224                                    #Set input size of each image

        elif self.backbone == "ConvNeXt":
            """ ConvNeXt tiny acc@1 (on ImageNet-1K): 82.52
            """
            model_ft = models.convnext_tiny(weights=model_weights)
            num_ftrs = model_ft.classifier[2].in_features
            model_ft.classifier[2] = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "EfficientNetV2":
            """ EfficientNetV2 small acc@1 (on ImageNet-1K): 84.228
            """
            model_ft = models.efficientnet_v2_s(weights=model_weights)
            num_ftrs = model_ft.classifier[1].in_features
            model_ft.classifier[1] = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 384

        elif self.backbone == "InceptionV3":
            """ InceptionV3 acc@1 (on ImageNet-1K): 77.294
            """
            model_ft = models.inception_v3(weights=model_weights)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 299

        elif self.backbone == "MaxVit":
            """ MaxVit acc@1 (on ImageNet-1K): 83.7
            """
            model_ft = models.maxvit_t(weights=model_weights)
            num_ftrs = model_ft.classifier[5].in_features
            model_ft.classifier[5] = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "MobileNetV3":
            """ MobileNet V3 acc@1 (on ImageNet-1K): 67.668
            """
            model_ft = models.mobilenet_v3_small(weights=model_weights)
            num_ftrs = model_ft.classifier[3].in_features
            model_ft.classifier[3] = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "ResNetXt":
            """ ResNeXt-50 32x4d acc@1 (on ImageNet-1K): 77.618
            """
            model_ft = models.resnext50_32x4d(weights=model_weights)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224


        elif self.backbone == "ShuffleNetV2":
            """ ShuffleNetV2 acc@1 (on ImageNet-1K): 60.552
            """
            model_ft = models.shufflenet_v2_x0_5(weights=model_weights)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "ViT":
            """ Vision Transformer acc@1 (on ImageNet-1K): 81.072
            """
            model_ft = models.vit_b_16(weights=model_weights)
            num_ftrs = model_ft.heads.head.in_features
            model_ft.heads.head = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "WideResNet":
            """ Wide ResNet acc@1 (on ImageNet-1K): 78.468
            """
            model_ft = models.wide_resnet50_2(weights=model_weights)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "densenet": 
            """ Densenet acc@1 (on ImageNet-1K): 74.434
            """
            model_ft = models.densenet121(weights=model_weights)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224
        
        elif self.backbone == "swin": 
            """ Swin Transformer tiny acc@1 (on ImageNet-1K): 81.474
            """
            model_ft = models.swin_t(weights=model_weights)
            num_ftrs = model_ft.head.in_features
            model_ft.head = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size

    def forward(self, tissue_tiles):

        latent_space = self.encoder(tissue_tiles)

        return latent_space


class contrastiveModel(torch.nn.Module):
    def __init__(self, ae_layer_list, ae_act, ae_pretrained_path, img_backbone, img_use_pretrained):
        """
        Contrastive model to do transcriptomic prediction and contrastive learning from images and genetic expression.

        Args:
            ae_layer_list (list): List of ints with the autoencoder layer dimensions.
            ae_act (str): String of coding for the activations according to https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
            ae_pretrained_path (str): Path containing the pretrained autoencoder model. "None" to train from scratch.
            img_backbone (str): Name of the backbone to use for embed images. Options are "resnet", "alexnet", "vgg", "squeezenet", "densenet".
            img_use_pretrained (bool): Boolean inidicating to use a pretrained image encoder over ImageNet 1k
        """
        super(contrastiveModel, self).__init__()
        self.ae_encoder_dims = ae_layer_list
        self.ae_decoder_dims = ae_layer_list[::-1]

        self.ae_act = ae_act
        self.ae_pretrained_path = ae_pretrained_path
        self.img_backbone = img_backbone
        self.img_use_pretrained = img_use_pretrained

        # Submodule declaration
        self.ae_encoder = MLP(self.ae_encoder_dims, self.ae_act)
        self.ae_decoder = MLP(self.ae_decoder_dims, self.ae_act)

        # Declare image encoder
        self.img_encoder = ImageEncoder(self.img_backbone, self.img_use_pretrained, self.ae_encoder_dims[-1])

        # If ae_pretrained is not 'None', then load the autoencoder and initialize the encoder and decoder with this weights
        if self.ae_pretrained_path != 'None':
            
            # Declare blank auto encoder
            pretrained_ae = autoEncoder(self.ae_encoder_dims, self.ae_act)
            
            # Load weights
            pretrained_ae.load_state_dict(torch.load(self.ae_pretrained_path))
            
            # Split autoencoder in pretrained encoder and decoder
            pretrained_encoder = pretrained_ae.encoder
            pretrained_decoder = pretrained_ae.decoder

            # Load weights in the encoder and decoder of the self class
            self.ae_encoder.load_state_dict(pretrained_encoder.state_dict())
            self.ae_decoder.load_state_dict(pretrained_decoder.state_dict())

    def load_img_encoder(self, path, freeze = False):

        base_stnet = STNet(self.img_backbone, self.img_use_pretrained, self.ae_encoder_dims[0])
        base_stnet.load_state_dict(torch.load(path))
        # FIXME: The change of the last layer will only work for densenet 121
        num_ftrs = base_stnet.encoder.encoder.classifier.in_features
        base_stnet.encoder.encoder.classifier = nn.Linear(num_ftrs, self.ae_encoder_dims[-1])

        self.img_encoder = base_stnet
        print(f'Loaded an ST-Net model from {path}')

    def freeze_img_encoder(self):
        for param in self.img_encoder.parameters():
            param.requires_grad = False
        # let the last layer trainable because of the change of output
        num_ftrs = self.img_encoder.encoder.encoder.classifier.in_features
        self.img_encoder.encoder.encoder.classifier = nn.Linear(num_ftrs, self.ae_encoder_dims[-1])
        print('Image encoder is frozen...')
    
    def freeze_exp_encoder(self):
        for param in self.ae_encoder.parameters():
            param.requires_grad = False
        print('Image encoder is frozen...')


    def forward(self, expression, tissue_tiles):
        
        # Get latents
        ae_latent = self.ae_encoder(expression)
        img_latent = self.img_encoder(tissue_tiles)

        # Get reconstructions
        ae_reconstruction = self.ae_decoder(ae_latent)
        img_reconstruction = self.ae_decoder(img_latent)

        return ae_reconstruction, img_reconstruction, ae_latent, img_latent


class STNet(torch.nn.Module):
    def __init__(self, backbone, use_pretrained,  output_dim):
        
        super(STNet, self).__init__()
        self.backbone = backbone
        self.use_pretrained = use_pretrained
        self.output_dim = output_dim
        self.input_size = 224

        self.encoder = ImageEncoder(self.backbone, self.use_pretrained, self.output_dim)
    
    def forward(self, tissue_tiles):

        out = self.encoder(tissue_tiles)

        return out


### Test code to vaidate the constructor of all the image encoders
if __name__ == "__main__":

    test_model_list = ['resnet', 'ConvNeXt', 'MobileNetV3', 'ResNetXt', 'ShuffleNetV2', 'ViT', 'WideResNet', 'densenet', 'swin']
    for model in test_model_list:
        print(f'Testing {model}...')
        try:
            test_model = ImageEncoder(backbone=model, use_pretrained=True,  latent_dim=256)
            test_model(torch.rand(1, 3, 224, 224))
            print(f'Passed {model} test')
        except:
            print(f'Failed {model} test')

