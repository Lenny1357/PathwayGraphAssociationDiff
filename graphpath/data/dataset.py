import networkx as nx
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


# Choice wether to ignore subsystem grouping
class PathwayGraphDataset(Dataset):
    """Convert edges DataFrame to graph samples for diffusion model with chunking support"""
    
    def __init__(self, node_id_map, graphs, max_nodes=64, device='cuda'):
        """
        Args:
            node_id_map: Dictionary mapping nodes to IDs
            graphs: List of graph samples
            max_nodes: maximum nodes per graph chunk
            device: torch device
        """
        self.max_nodes = max_nodes
        self.device = device
        self.graphs = graphs

        # Build vocabulary for all unique nodes
        self.all_nodes = list(node_id_map.keys())
        self.node_to_id = node_id_map

        self.id_to_node = {idx: node for node, idx in self.node_to_id.items()}
        self.vocab_size = len(self.node_to_id)
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        graph = self.graphs[idx]
        
        # Get node IDs (indices into vocabulary)
        node_ids = torch.tensor(graph['node_ids'], dtype=torch.long)
        
        # Get adjacency matrix
        adj = torch.tensor(graph['adj_matrix'], dtype=torch.float32)
        
        # Pad to max_nodes
        n = graph['num_nodes']
        if n < self.max_nodes:
            # Pad adjacency matrix
            adj_padded = torch.zeros(self.max_nodes, self.max_nodes)
            adj_padded[:n, :n] = adj
            
            # Pad node IDs (use 0 for padding)
            node_ids_padded = torch.zeros(self.max_nodes, dtype=torch.long)
            node_ids_padded[:n] = node_ids
        else:
            adj_padded = adj
            node_ids_padded = node_ids
        
        # Create node mask (1 for real nodes, 0 for padding)
        node_mask = torch.zeros(self.max_nodes)
        node_mask[:n] = 1
        
        return adj_padded, node_ids_padded, node_mask, graph['original_group_id'], graph['center_node']