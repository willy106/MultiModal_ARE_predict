import torch
import torch.nn as nn
import torch.nn.functional as F
from .Convnext3D import ConvNeXt3DEncoder


class TableDataEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TableDataEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

# Combined Model
class CombinedModel(nn.Module):
    def __init__(self, image_input_channels, image_num_blocks, image_dim, table_input_dim, table_hidden_dim):
        super(CombinedModel, self).__init__()
        self.image_encoder = ConvNeXt3DEncoder(image_input_channels, image_num_blocks, image_dim)
        self.table_encoder = TableDataEncoder(table_input_dim, table_hidden_dim)
        self.fc1 = nn.Linear(image_dim+table_hidden_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, image, table):
        image_features = self.image_encoder(image)
        table_features = self.table_encoder(table)
        table_features = torch.squeeze(table_features, 1)
        combined = torch.cat((image_features, table_features), dim=1)
        x = F.relu(self.fc1(combined))
        x=self.fc2(x)
        return x