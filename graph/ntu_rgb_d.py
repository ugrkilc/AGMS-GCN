import numpy as np
import graph.tools as tl
import sys
sys.path.extend(['../'])

class Graph:
    def __init__(self,num_node):
        self.num_node = num_node    
        self.hop_size = 2     
        self.get_edge()
        self.hop_dis = tl.get_hop_distance(self.num_node, self.edge,self.hop_size)
        self.get_adjacency()

    def __str__(self):
        return self.A

    def create_edges(self, neighbor_base):
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_base]
        return self_link + neighbor_link
    
    def get_edge(self):   
        if self.num_node == 25:           
            neighbor_base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                             (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                             (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                             (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                             (22, 23), (23, 8), (24, 25), (25, 12)]
        elif self.num_node == 11:           
            neighbor_base = [(1, 2), (2, 3), (2, 4), (4, 5), (2, 6), (6, 7),
                             (1, 8), (8, 9), (1, 10), (10, 11)]
        elif self.num_node == 6:               
            neighbor_base = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 3),
                             (2, 4), (2, 5), (2, 6), (3, 6), (4, 5), (5, 6)]
        else:
            raise ValueError('Do not exist this kind of graph')

        self.edge = self.create_edges(neighbor_base)
        self.center = 1



    def get_adjacency(self):
        valid_hop = range(0, self.hop_size + 1, 1)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = tl.normalize_digraph(adjacency)

        A = []
        for hop in valid_hop:
            a_root = np.zeros((self.num_node, self.num_node))
            a_close = np.zeros((self.num_node, self.num_node))
            a_further = np.zeros((self.num_node, self.num_node))
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if self.hop_dis[j, i] == hop:
                        if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                            a_root[j, i] = normalize_adjacency[j, i]
                        elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                            a_close[j, i] = normalize_adjacency[j, i]
                        else:
                            a_further[j, i] = normalize_adjacency[j, i]
            if hop == 0:
                A.append(a_root)
            else:
                A.append(a_root + a_close)
                A.append(a_further)
        A = np.stack(A)
        self.A = A







