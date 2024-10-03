import torch
import torch.nn as nn
from aamg_module import AAMG
from gfe_module import GFE_one, GFE_two, GFE_three,GFE_one_temp, GFE_two_temp

import graph.ntu_rgb_d as Graph

def import_class(name):    
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)        
    return mod

class Model(nn.Module):
    def __init__(self, num_classes,residual, dropout, num_person, graph, num_nodes,input_channels):
        super().__init__()
        
        # Initialize graph
        self._initialize_graph(graph, num_nodes)      
      
        A_size = self.A.size()
        # Initialize layers
        self._initialize_layers(num_person, num_classes, residual, dropout, A_size, num_nodes, input_channels)

    def _initialize_graph(self, graph, num_nodes):
        if graph is None:
            raise ValueError("Graph cannot be None")
        else:
            Graph = import_class(graph)
            self.graph_instance = Graph(num_nodes)
        
        A = torch.tensor(self.graph_instance.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

    def _initialize_layers(self, num_person, num_classes, residual, dropout, A_size, num_nodes, input_channels):    

        self.gfe_one = GFE_one(num_person,num_classes, dropout, residual, A_size, input_channels)
        self.gfe_one_temp = GFE_one_temp(num_person,num_classes, dropout, residual, A_size)
        self.aamg_one = AAMG(128, num_classes, num_nodes)

        self.gfe_two = GFE_two(num_person,num_classes, dropout, residual, A_size)
        self.gfe_two_temp = GFE_two_temp(num_person,num_classes, dropout, residual, A_size)
        self.aamg_two = AAMG(256, num_classes, num_nodes)

        self.gfe_three = GFE_three(num_person,num_classes, dropout, residual, A_size)

    def forward(self, x):
  
        
        feature_map_one = self.gfe_one(x, self.A)    
        feature_map_one_temp = self.gfe_one_temp(feature_map_one, self.A)
        attention_matrices_one = self.aamg_one(feature_map_one_temp)

        feature_map_two, output_two = self.gfe_two(feature_map_one, self.A, attention_matrices_one)
        feature_map_two_temp = self.gfe_two_temp(feature_map_two, self.A)
        attention_matrices_two = self.aamg_two(feature_map_two_temp)

        _, output_three = self.gfe_three(feature_map_two, self.A, attention_matrices_two)

        # return output_two, output_three, attention_matrices_one, attention_matrices_two
        return output_two, output_three
    
