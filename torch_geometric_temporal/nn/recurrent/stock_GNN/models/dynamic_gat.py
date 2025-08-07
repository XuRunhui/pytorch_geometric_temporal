from torch_geometric_temporal.nn.attention.astgcn import ASTGCN
from torch_geometric_temporal.nn.recurrent import A3TGCN2
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops
import torch
import torch.nn as nn
import torch.nn.functional as F


class Dynamic_Gat(nn.Module):
    def __init__(
            self, gru_params: dict, 
            astgcn_params: dict, 
            k_nn: int = 8, 
            add_self_loops: bool = True, 
            linear_output_dim: int = 32,
            predict_return: bool = False,
            resnet_blocks: int = 2,
            resnet_hidden_dim: int = 32
        ):
        """ 
        Initialize the Dynamic_Gat model with ASTGCN and GRU parameters.

        Args:
            gru_params (dict): Dictionary containing parameters for the GRU.
            astgcn_params (dict): Dictionary containing parameters for the ASTGCN model.
            k_nn (int): Number of nearest neighbors for dynamic graph construction.
            add_self_loops (bool): Whether to add self-loops to the graph.
            resnet_blocks (int): Number of ResNet blocks after GRU.
            resnet_hidden_dim (int): Hidden dimension for ResNet blocks.
        """
        super(Dynamic_Gat, self).__init__()

        # Initialize the ASTGCN model using the parameters from the dictionary
        self.gat = A3TGCN2(**astgcn_params)

        # Initialize GRU
        self.gru = nn.GRU(**gru_params)

        # Parameters for dynamic graph construction
        self.k_nn = k_nn
        self.add_self_loops = add_self_loops
        self.linear_output_dim = linear_output_dim
        self.predict_return = predict_return
        self.resnet_blocks = resnet_blocks
        self.resnet_hidden_dim = resnet_hidden_dim

        # Get GRU hidden dimension
        self.gru_hidden_dim = gru_params["hidden_size"]
        
        # ResNet blocks for processing GRU output
        self.resnet_layers = nn.ModuleList()
        # self.batch_norm = nn.BatchNorm1d(self.final_gnn_dim)
        # First layer to transform GRU output to ResNet hidden dimension
        self.input_projection = nn.Linear(self.gru_hidden_dim, self.resnet_hidden_dim)
        self.batch_norm = nn.BatchNorm1d(self.resnet_hidden_dim)
        # ResNet blocks
        for _ in range(self.resnet_blocks):
            self.resnet_layers.append(self._make_resnet_block(self.resnet_hidden_dim))
        
        # Final fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.resnet_hidden_dim, self.resnet_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.resnet_hidden_dim // 2, self.gru_hidden_dim),  # Back to original GRU dimension
            nn.LayerNorm(self.gru_hidden_dim)
        )

        self.linear = nn.Linear(astgcn_params["out_channels"], self.linear_output_dim)

        if self.predict_return:
            print("🎯 Building return prediction layers")
            self.return_predictor = nn.ModuleList([
                nn.ReLU(),
                nn.Linear(self.linear_output_dim, 7)
            ])

    def _make_resnet_block(self, hidden_dim):
        """Create a ResNet block"""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def _apply_resnet_block(self, x, resnet_block):
        """Apply ResNet block with residual connection"""
        residual = x
        out = resnet_block(x)
        return F.relu(out + residual)  # Residual connection

    
    def construct_edge(self, x_seq):
        # Handle input shape
        if x_seq.dim() == 3:
            b, n, f = x_seq.shape
            l = 1
            x_seq = x_seq.unsqueeze(1)
        else:
            b, l, f, n = x_seq.shape
        
        # Dual GRU encoding
        gru_in = x_seq.permute(1, 0, 3, 2).reshape(l, b * n, f)
        
        # Similarity GRU for graph construction
        gru_out_sim, _ = self.gru(gru_in)
        h = gru_out_sim[-1].view(b, n, -1)  # [b, n, gru_hidden_dim]
        
        # Apply ResNet blocks and fully connected layers to enhance GRU output
        # Project to ResNet hidden dimension
        h_projected = self.input_projection(h)  # [b, n, resnet_hidden_dim]
        
        # # Apply ResNet blocks
        h_resnet = h_projected
        # for resnet_block in self.resnet_layers:
        #     h_resnet = self._apply_resnet_block(h_resnet, resnet_block)
        
        # # Apply final fully connected layers
        h_resnet = self.batch_norm(h_resnet)
        h_processed = self.fc_layers(h_resnet)  # [b, n, gru_hidden_dim]
        # h_processed = h
        # Dynamic graph construction using processed features
        sim = torch.einsum("bni,bmi->bnm", h, h)  # [b, n, n]
        
        # Top-k edge selection
        sim_masked = sim.clone()
        eye_mask = torch.eye(n, device=sim.device).bool().unsqueeze(0).expand(b, -1, -1)
        sim_masked[eye_mask] = -1e9
        
        topk_vals, topk_idx = sim_masked.topk(self.k_nn, dim=-1, sorted=True)
        
        # Create edge indices
        edge_src = torch.arange(n, device=sim.device).unsqueeze(0).unsqueeze(2).expand(b, n, self.k_nn).contiguous()
        edge_dst = topk_idx.contiguous()
        edge_weight = topk_vals.contiguous()
        
        # Create batch data for GNN
        data_list = []
        for i in range(b):
            src_flat = edge_src[i].contiguous().view(-1)
            dst_flat = edge_dst[i].contiguous().view(-1)
            weight_flat = edge_weight[i].contiguous().view(-1)
            
            data = Data(
                x=h_processed[i],  # Use processed features instead of raw GRU output
                edge_index=torch.stack([src_flat, dst_flat], dim=0),
                edge_weight=weight_flat
            )
            data_list.append(data)
        
        batch_data = Batch.from_data_list(data_list)
        x_all = batch_data.x
        e_idx = batch_data.edge_index
        e_w = batch_data.edge_weight
        
        # Add self loops for GCN
        if self.add_self_loops:
            e_idx, e_w = add_self_loops(
                batch_data.edge_index,
                batch_data.edge_weight,
                fill_value=1.0,
                num_nodes=batch_data.num_nodes
            )
        
        return e_idx, e_w, h_processed  # Return processed features

    
    def forward_return(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through return prediction layers"""
        for layer in self.return_predictor:
            x = layer(x)
        return x
    
    
    def forward(self, x, edge_index=None):
        """
        Forward pass for the Dynamic_Gat model.

        Args:
            b, l, f, n
            x (torch.Tensor): Input node features of shape (B, N, F_in, T_in).
            edge_index (torch.Tensor, optional): Predefined edge indices of the graph. If None, it will be dynamically computed.

        Returns:
            torch.Tensor: Output predictions of shape (B, N, T_out).
        """
       
        edge_index, edge_weight, gru_h = self.construct_edge(x)

        gat_in = x.permute(0,3,2,1)
        # print(f'input for gat {gat_in.shape}')
        gat_out = self.gat(gat_in, edge_index, edge_weight) # x [b, 207, 2, 12]  returns h [b, 207, 32]

        relu_result = F.relu(gat_out)

        final_out = self.linear(relu_result)

        if self.predict_return:
            final_out = self.forward_return(final_out)
        print(gru_h.shape)
        return final_out, gru_h


    def get_model_stats(self) -> dict:
        """Get model statistics for logging"""
        stats = {
            'node_feat_dim': 24,
            'gru_hidden_dim': self.gru_hidden_dim,
            'resnet_blocks': self.resnet_blocks,
            'resnet_hidden_dim': self.resnet_hidden_dim,
            'output_factor_dim': self.linear_output_dim,
            'pure_gru': False,
            'gnn_type': 'gcn',
            'k_nn': self.k_nn,
            'predict_return': self.predict_return,
            'has_resnet': True,
            'has_fc_layers': True
        }
        
        return stats