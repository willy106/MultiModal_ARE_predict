import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.gated_tab_transformer import GatedTabTransformer
from .Convnext3D import ConvNeXt3DEncoder


# class TableDataEncoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(TableDataEncoder, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return x


# model = GatedTabTransformer(
#     categories = (2, 2, 2, 2),      # tuple containing the number of unique values within each category
#     num_continuous = 10,                # number of continuous values
#     transformer_dim = 32,               # dimension, paper set at 32
#     dim_out = table_hidden_dim,                        # binary prediction, but could be anything
#     transformer_depth = 6,              # depth, paper recommended 6
#     transformer_heads = 8,              # heads, paper recommends 8
#     attn_dropout = 0.1,                 # post-attention dropout
#     ff_dropout = 0.1,                   # feed forward dropout
#     mlp_act = nn.LeakyReLU(0),          # activation for final mlp, defaults to relu, but could be anything else (selu, etc.)
#     mlp_depth=4,                        # mlp hidden layers depth
#     mlp_dimension=32,                   # dimension of mlp layers
#     gmlp_enabled=True                   # gmlp or standard mlp
# )

# Combined Model
class CombinedModel(nn.Module):
    def __init__(self, image_input_channels, image_num_blocks, image_dim,table_hidden_dim):
        super(CombinedModel, self).__init__()
        self.image_encoder = ConvNeXt3DEncoder(image_input_channels, image_num_blocks, image_dim)
        self.table_encoder = GatedTabTransformer(
                                categories = (2, 2, 2, 2),      # tuple containing the number of unique values within each category
                                num_continuous = 10,                # number of continuous values
                                transformer_dim = 32,               # dimension, paper set at 32
                                dim_out = table_hidden_dim,                        # binary prediction, but could be anything
                                transformer_depth = 6,              # depth, paper recommended 6
                                transformer_heads = 8,              # heads, paper recommends 8
                                attn_dropout = 0.1,                 # post-attention dropout
                                ff_dropout = 0.1,                   # feed forward dropout
                                mlp_act = nn.LeakyReLU(0),          # activation for final mlp, defaults to relu, but could be anything else (selu, etc.)
                                mlp_depth=4,                        # mlp hidden layers depth
                                mlp_dimension=32,                   # dimension of mlp layers
                                gmlp_enabled=True                   # gmlp or standard mlp
        )

        self.fc1 = nn.Linear(image_dim+table_hidden_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, image, category_features,continuous_feature):
        image_features = self.image_encoder(image)
        table_features = self.table_encoder(category_features,continuous_feature)
        table_features = torch.squeeze(table_features, 1)
        combined = torch.cat((image_features, table_features), dim=1)
        x = F.relu(self.fc1(combined))
        x=self.fc2(x)
        return x