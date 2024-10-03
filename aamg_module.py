import torch
import torch.nn as nn
import torch.nn.functional as F

class AAMG(nn.Module):
    def __init__(self, input_channels, num_classes, num_nodes):
        super().__init__()
        self.num_attention_matrices = 5
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, num_classes, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_classes)
        )
        
        self.attention_extractor = nn.Sequential(
            nn.Conv2d(num_classes, self.num_attention_matrices * num_nodes, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.num_attention_matrices * num_nodes),       
        )

    def forward(self, feature_map):
        NM, _, _, V = feature_map.size()
        
        feature_map = self.feature_extractor(feature_map)
        feature_map = F.avg_pool2d(feature_map, (feature_map.size()[2], 1))  
        feature_map = self.attention_extractor(feature_map)
        
        attention_matrices = feature_map.view(NM, self.num_attention_matrices, V, V)
        attention_matrices = F.relu(attention_matrices)

        threshold = attention_matrices.mean()
        attention_matrices = torch.where(attention_matrices < threshold, -attention_matrices, attention_matrices)
        
        return F.relu(attention_matrices)
