# models/dyn_graph_module.py
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from torch_geometric.nn import GCNConv  # 或者 GATConv, GraphConv…
from torch_geometric.data import Batch
from torch_geometric_temporal.nn.recurrent.adp_adj_loss import AccumulativeGainLoss  # 假设你有这个自定义损失函数
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops

class DynamicGraphLightning(pl.LightningModule):
    def __init__(
        self,
        node_feat_dim: int,
        gru_hidden_dim: int = 64,
        gnn_hidden_dim: int = 64,
        k_nn: int = 8,
        lr: float = 1e-3,
        loss_fn: nn.Module = AccumulativeGainLoss(value_decay=0.9, penalty_weight=0.1, eps=1e-8),
        add_self_loops: bool = True,
    ):
        super().__init__()
        # 1) 用于序列编码的 GRU（或 LSTM/GRUCell）
        self.gru = nn.GRU(
            input_size=node_feat_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
        )
        # 2) 用于图消息传递的 GNN 层
        self.gnn1 = GCNConv(gru_hidden_dim, gnn_hidden_dim, add_self_loops=add_self_loops)
        self.gnn2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim, add_self_loops=add_self_loops)
        # 3) 最后预测层
        self.predictor = nn.Linear(gnn_hidden_dim, 1)  # 回归示例

        self.k_nn = k_nn
        self.lr = lr
        self.loss_fn = loss_fn
        self.add_self_loops = add_self_loops

    def forward(self, data_input) -> list:
        """
        data_input: 可以是以下两种格式之一:
        1. torch.Tensor: [batch_size, num_nodes, seq_len, node_feat_dim] - 所有图大小相同
        2. list[torch.Tensor]: [graph1_seq, graph2_seq, ...] - 每个图可以有不同大小
           其中每个graph_seq的形状为: [num_nodes_i, seq_len, node_feat_dim]
        
        返回：
        1. 如果输入是tensor: torch.Tensor [batch_size, num_nodes, 1]
        2. 如果输入是list: list[torch.Tensor] [graph1_output, graph2_output, ...]
        """
        if isinstance(data_input, torch.Tensor):
            # 处理固定大小的批量图数据
            return self._forward_batch_tensor(data_input)
        elif isinstance(data_input, list):
            # 处理不同大小的图数据列表
            return self._forward_graph_list(data_input)
        else:
            raise ValueError("Input must be either torch.Tensor or list of torch.Tensor")
    
    def _forward_batch_tensor(self, x_seq: torch.Tensor) -> torch.Tensor:
        """处理固定大小的批量图数据"""
        # ——— 1. GRU 编码 ——————————————
        if x_seq.dim() == 3:
            # If input is [batch_size, num_nodes, feat_dim], add seq_len dimension
            b, n, f = x_seq.shape
            l = 1
            x_seq = x_seq.unsqueeze(2)  # [batch_size, num_nodes, 1, feat_dim]
        else:
            b, n, l, f = x_seq.shape
        
        # Reshape for GRU: [seq_len, batch_size * num_nodes, feat_dim]
        gru_in = x_seq.permute(2, 0, 1, 3).reshape(l, b * n, f)
        gru_out, _ = self.gru(gru_in)    # 输出: [seq_len, batch_size * num_nodes, gru_hidden_dim]

        h = gru_out[-1].view(b, n, -1)  # 取最后一个时间步的输出: [batch_size, num_nodes, gru_hidden_dim]
        # TensorBoard logging: log GRU output stats
        if hasattr(self, 'logger') and self.logger is not None and hasattr(self.logger, 'experiment') and self.global_rank == 0:
            self.logger.experiment.add_scalar('stats/gru_output_mean', h.mean().item(), self.global_step)
            self.logger.experiment.add_scalar('stats/gru_output_std', h.std().item(), self.global_step)
        # ——— 2. 动态构图（相似度 + Top-k） —————
        # 计算节点两两点积
        sim = torch.einsum("bni,bmi->bnm", h, h)  # [b, n, n]
        # TensorBoard logging: log similarity stats
        if hasattr(self, 'logger') and self.logger is not None and hasattr(self.logger, 'experiment') and self.global_rank == 0:
            self.logger.experiment.add_scalar('stats/sim_mean', sim.mean().item(), self.global_step)
            self.logger.experiment.add_scalar('stats/sim_std', sim.std().item(), self.global_step)
        # Top-k
        topk_vals, topk_idx = sim.topk(self.k_nn + 1, dim=-1)  # 包含自己
        # 去掉自环
        node_idx = torch.arange(n, device=sim.device)[None, :, None]
        node_idx_k1 = node_idx.view(1, n, 1).expand(b, n, self.k_nn+1)  # [b,n,k+1]

        # 3) mask 去掉自循环 (i != i)
        mask = topk_idx != node_idx_k1                         # [b,n,k+1]

        # 4) 用同一个 mask 索引，留下 k 条真正的邻居边
        edge_weight = topk_vals[mask].view(b, n, self.k_nn)    # [b, n, k]
        edge_dst    = topk_idx[mask].view(b, n, self.k_nn)     # [b, n, k]
        edge_src    = node_idx_k1[mask].view(b, n, self.k_nn)  # [b, n, k]

        data_list = []
        for i in range(b):
            # Flatten edge indices and weights for this batch item
            src_flat = edge_src[i].view(-1)  # [n*k]
            dst_flat = edge_dst[i].view(-1)  # [n*k]
            weight_flat = edge_weight[i].view(-1)  # [n*k]
            
            data = Data(
                x=h[i],  # [n, feat_dim]
                edge_index=torch.stack([src_flat, dst_flat], dim=0),  # [2, n*k]
                edge_weight=weight_flat  # [n*k]
            )
            data_list.append(data)
        # # 把 batch 展平成一个大图: 假设 Lightning 的 batch_size=1，或自行处理
        # edge_index = torch.stack([edge_src, edge_dst], dim=0)  # [2, n*k] or [2, b*n*k]

        # # ——— 3. GNN 消息传递 ——————————————
        # # 把节点特征 flatten 成 [N_total, feat]
        # x = h.view(-1, h.size(-1))
        # e_idx = edge_index.view(2, -1)
        # e_w   = edge_weight.view(-1)
        batch_data = Batch.from_data_list(data_list)
        x_all    = batch_data.x            # [b*n, feat_dim]
        e_idx    = batch_data.edge_index   # [2, b*n*k]
        e_w      = batch_data.edge_weight  # [b*n*k]
        if(self.add_self_loops):
            e_idx, e_w = add_self_loops(
                batch_data.edge_index,
                batch_data.edge_weight,
                fill_value=1.0,       # 或其他你想给自环的权重
                num_nodes=batch_data.num_nodes
            )
        # 第一层
        out1 = F.relu(self.gnn1(x_all, e_idx, edge_weight=e_w))
        # 第二层
        out2 = F.relu(self.gnn2(out1, e_idx, edge_weight=e_w)) #[b*n, feat_dim]

        # ——— 4. 预测输出 ——————————————
        out = self.predictor(out2)  # [N_total, 1]
        final_out = out.view(b, n, -1)  # [batch, num_nodes, 1]
        # TensorBoard logging: log output stats
        if hasattr(self, 'logger') and self.logger is not None and hasattr(self.logger, 'experiment') and self.global_rank == 0:
            self.logger.experiment.add_scalar('stats/final_output_mean', final_out.mean().item(), self.global_step)
            self.logger.experiment.add_scalar('stats/final_output_std', final_out.std().item(), self.global_step)
        return final_out
    
    def _forward_graph_list(self, graph_list: list) -> list:
        """处理不同大小的图数据列表"""
        results = []
        
        for i, graph_seq in enumerate(graph_list):
            # 每个图独立处理: [num_nodes_i, seq_len, node_feat_dim]
            if graph_seq.dim() == 2:
                # 如果是 [num_nodes, feat_dim]，添加seq_len维度
                n, f = graph_seq.shape
                l = 1
                graph_seq = graph_seq.unsqueeze(1)  # [num_nodes, 1, feat_dim]
            else:
                n, l, f = graph_seq.shape
            
            # TensorBoard logging for individual graphs
            if hasattr(self, 'logger') and self.logger is not None and hasattr(self.logger, 'experiment') and self.global_rank == 0:
                self.logger.experiment.add_scalar(f'graph_stats/graph_{i}_num_nodes', n, self.global_step)
            
            # ——— 1. GRU 编码 ——————————————
            # Reshape for GRU: [seq_len, num_nodes, feat_dim]
            gru_in = graph_seq.permute(1, 0, 2)  # [seq_len, num_nodes, feat_dim]
            gru_out, _ = self.gru(gru_in)    # 输出: [seq_len, num_nodes, gru_hidden_dim]
            
            h = gru_out[-1]  # 取最后一个时间步: [num_nodes, gru_hidden_dim]
            
            # TensorBoard logging: log GRU output stats for this graph
            if hasattr(self, 'logger') and self.logger is not None and hasattr(self.logger, 'experiment') and self.global_rank == 0:
                self.logger.experiment.add_scalar(f'graph_stats/graph_{i}_gru_output_mean', h.mean().item(), self.global_step)
                self.logger.experiment.add_scalar(f'graph_stats/graph_{i}_gru_output_std', h.std().item(), self.global_step)
            
            # ——— 2. 动态构图（相似度 + Top-k） —————
            # 计算节点两两点积 [num_nodes, num_nodes]
            sim = torch.einsum("ni,mi->nm", h, h)
            
            # TensorBoard logging: log similarity stats for this graph
            if hasattr(self, 'logger') and self.logger is not None and hasattr(self.logger, 'experiment') and self.global_rank == 0:
                self.logger.experiment.add_scalar(f'graph_stats/graph_{i}_sim_mean', sim.mean().item(), self.global_step)
                self.logger.experiment.add_scalar(f'graph_stats/graph_{i}_sim_std', sim.std().item(), self.global_step)
            
            # 确保k_nn不超过节点数-1
            k = min(self.k_nn, n - 1)
            if k <= 0:
                # 如果图太小，直接用线性层处理
                out = self.predictor(h)  # [num_nodes, 1]
                results.append(out)
                continue
            
            # Top-k选择
            topk_vals, topk_idx = sim.topk(k + 1, dim=-1)  # 包含自己
            
            # 去掉自环
            node_idx = torch.arange(n, device=sim.device)[:, None]  # [n, 1]
            mask = topk_idx != node_idx  # [n, k+1]
            
            # 提取前k个邻居（排除自己）
            edge_weights = []
            edge_sources = []
            edge_targets = []
            
            for node in range(n):
                valid_neighbors = topk_idx[node][mask[node]][:k]  # 取前k个
                valid_weights = topk_vals[node][mask[node]][:k]
                
                # 添加边
                edge_sources.extend([node] * len(valid_neighbors))
                edge_targets.extend(valid_neighbors.tolist())
                edge_weights.extend(valid_weights.tolist())
            
            if len(edge_sources) == 0:
                # 如果没有边，直接用线性层
                out = self.predictor(h)
                results.append(out)
                continue
            
            # 构建图数据
            edge_index = torch.tensor([edge_sources, edge_targets], 
                                    dtype=torch.long, device=h.device)
            edge_weight = torch.tensor(edge_weights, dtype=torch.float, device=h.device)
            
            # 添加自环（如果需要）
            if self.add_self_loops:
                edge_index, edge_weight = add_self_loops(
                    edge_index, edge_weight,
                    fill_value=1.0, num_nodes=n
                )
            
            # ——— 3. GNN 消息传递 ——————————————
            out1 = F.relu(self.gnn1(h, edge_index, edge_weight=edge_weight))
            out2 = F.relu(self.gnn2(out1, edge_index, edge_weight=edge_weight))
            
            # ——— 4. 预测输出 ——————————————
            out = self.predictor(out2)  # [num_nodes, 1]
            
            # TensorBoard logging: log output stats for this graph
            if hasattr(self, 'logger') and self.logger is not None and hasattr(self.logger, 'experiment') and self.global_rank == 0:
                self.logger.experiment.add_scalar(f'graph_stats/graph_{i}_output_mean', out.mean().item(), self.global_step)
                self.logger.experiment.add_scalar(f'graph_stats/graph_{i}_output_std', out.std().item(), self.global_step)
            
            results.append(out)
        
        return results

    def training_step(self, batch, batch_idx):
        # batch: 来自你的 DataLoader，格式可以是 (x_t, y_t)
        x_t, y_t = batch
        y_pred = self(x_t)
        
        # 处理不同的输出格式
        if isinstance(y_pred, list):
            # 变长图的情况
            total_loss = 0
            total_nodes = 0
            for i, (pred, target) in enumerate(zip(y_pred, y_t)):
                if isinstance(target, torch.Tensor):
                    loss_i = self.loss_fn(pred.squeeze(-1), target.float())
                    total_loss += loss_i * pred.size(0)  # 按节点数加权
                    total_nodes += pred.size(0)
            loss = total_loss / total_nodes if total_nodes > 0 else total_loss
        else:
            # 固定大小图的情况
            loss = self.loss_fn(y_pred.squeeze(-1), y_t.float())
        
        # TensorBoard logging: log training loss
        if hasattr(self, 'logger') and self.logger is not None and hasattr(self.logger, 'experiment') and self.global_rank == 0:
            self.logger.experiment.add_scalar('loss/train', loss.item(), self.global_step)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_t, y_t = batch
        y_pred = self(x_t)
        
        # 处理不同的输出格式
        if isinstance(y_pred, list):
            # 变长图的情况
            total_loss = 0
            total_nodes = 0
            for i, (pred, target) in enumerate(zip(y_pred, y_t)):
                if isinstance(target, torch.Tensor):
                    loss_i = self.loss_fn(pred.squeeze(-1), target.float())
                    total_loss += loss_i * pred.size(0)  # 按节点数加权
                    total_nodes += pred.size(0)
            loss = total_loss / total_nodes if total_nodes > 0 else total_loss
        else:
            # 固定大小图的情况
            loss = self.loss_fn(y_pred.squeeze(-1), y_t.float())
        
        # TensorBoard logging: log validation loss
        if hasattr(self, 'logger') and self.logger is not None and hasattr(self.logger, 'experiment') and self.global_rank == 0:
            self.logger.experiment.add_scalar('loss/val', loss.item(), self.global_step)
        self.log("val/loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

