"""
Graph Attention Network (GAT) Model for Node-Centric AI System
================================================================
Advanced graph-based neural network for workflow next-node prediction.
This model treats workflows as directed graphs rather than sequences,
enabling better pattern learning from structural relationships.
"""
import math
from typing import Optional, Tuple, List, Dict
from collections import defaultdict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: NetworkX not available")

# Try to import PyTorch Geometric
try:
    from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("Warning: PyTorch Geometric not available. GAT model will be limited.")


if TORCH_AVAILABLE:
    
    class Node2VecEmbedding(nn.Module):
        """
        Node2Vec-style random walk embeddings for workflow graphs.
        Learns node representations based on graph structure.
        """
        
        def __init__(
            self,
            num_nodes: int,
            embed_dim: int = 64,
            walk_length: int = 10,
            num_walks: int = 20,
            p: float = 1.0,
            q: float = 1.0
        ):
            super().__init__()
            self.num_nodes = num_nodes
            self.embed_dim = embed_dim
            self.walk_length = walk_length
            self.num_walks = num_walks
            self.p = p  # Return parameter
            self.q = q  # In-out parameter
            
            # Learnable node embeddings
            self.embedding = nn.Embedding(num_nodes, embed_dim)
            nn.init.xavier_uniform_(self.embedding.weight)
        
        def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
            """Get embeddings for node IDs"""
            return self.embedding(node_ids)
        
        @staticmethod
        def random_walk(
            edge_index: torch.Tensor,
            start_node: int,
            walk_length: int,
            p: float = 1.0,
            q: float = 1.0
        ) -> List[int]:
            """
            Perform a single biased random walk from start_node.
            p: Return parameter (likelihood of returning to previous node)
            q: In-out parameter (differentiates inward vs outward nodes)
            """
            walk = [start_node]
            
            # Build adjacency list from edge_index
            adj = defaultdict(list)
            for i in range(edge_index.size(1)):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                adj[src].append(dst)
            
            for _ in range(walk_length - 1):
                curr = walk[-1]
                neighbors = adj.get(curr, [])
                
                if not neighbors:
                    break
                
                if len(walk) == 1:
                    # First step: uniform random
                    next_node = neighbors[torch.randint(len(neighbors), (1,)).item()]
                else:
                    prev = walk[-2]
                    # Compute transition probabilities
                    probs = []
                    for neighbor in neighbors:
                        if neighbor == prev:
                            probs.append(1.0 / p)
                        elif neighbor in adj.get(prev, []):
                            probs.append(1.0)
                        else:
                            probs.append(1.0 / q)
                    
                    # Normalize and sample
                    probs = torch.tensor(probs)
                    probs = probs / probs.sum()
                    idx = torch.multinomial(probs, 1).item()
                    next_node = neighbors[idx]
                
                walk.append(next_node)
            
            return walk


    class GraphAttentionLayer(nn.Module):
        """
        Single Graph Attention Layer (if PyG not available).
        Implements attention mechanism over graph neighborhoods.
        """
        
        def __init__(
            self,
            in_features: int,
            out_features: int,
            dropout: float = 0.2,
            alpha: float = 0.2,
            concat: bool = True
        ):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.alpha = alpha
            self.concat = concat
            
            # Learnable parameters
            self.W = nn.Linear(in_features, out_features, bias=False)
            self.a = nn.Parameter(torch.zeros(2 * out_features, 1))
            nn.init.xavier_uniform_(self.a)
            
            self.leaky_relu = nn.LeakyReLU(alpha)
            self.dropout = nn.Dropout(dropout)
        
        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor
        ) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: Node features (num_nodes, in_features)
                edge_index: Graph edges (2, num_edges)
            
            Returns:
                Updated node features (num_nodes, out_features)
            """
            # Linear transformation
            Wh = self.W(x)  # (N, out_features)
            
            # Compute attention coefficients
            src, dst = edge_index[0], edge_index[1]
            
            # Concatenate source and target features
            edge_features = torch.cat([Wh[src], Wh[dst]], dim=1)  # (E, 2*out_features)
            
            # Attention scores
            e = self.leaky_relu(edge_features @ self.a).squeeze(-1)  # (E,)
            
            # Softmax over neighborhoods
            attention = torch.zeros(x.size(0), x.size(0), device=x.device)
            attention[src, dst] = e
            
            attention = F.softmax(attention, dim=1)
            attention = self.dropout(attention)
            
            # Aggregate
            h_prime = attention @ Wh  # (N, out_features)
            
            if self.concat:
                return F.elu(h_prime)
            else:
                return h_prime


    class GraphAttentionNetwork(nn.Module):
        """
        Graph Attention Network for workflow next-node prediction.
        
        This model:
        1. Embeds node types as initial features
        2. Applies multi-head graph attention layers
        3. Aggregates graph representation
        4. Predicts next node from the vocabulary
        """
        
        def __init__(
            self,
            vocab_size: int,
            embed_dim: int = 64,
            hidden_dim: int = 128,
            num_heads: int = 4,
            num_layers: int = 2,
            dropout: float = 0.2,
            use_pyg: bool = True,
            pad_idx: int = 0
        ):
            super().__init__()
            
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.hidden_dim = hidden_dim
            self.num_heads = num_heads
            self.use_pyg = use_pyg and PYG_AVAILABLE
            
            # Node type embedding
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
            
            if self.use_pyg:
                # PyTorch Geometric GAT layers
                self.conv_layers = nn.ModuleList()
                
                # First layer: embed_dim -> hidden_dim
                self.conv_layers.append(
                    GATConv(embed_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
                )
                
                # Hidden layers
                for _ in range(num_layers - 1):
                    self.conv_layers.append(
                        GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
                    )
                
                self.output_dim = hidden_dim
            else:
                # Fallback: custom GAT layers
                self.conv_layers = nn.ModuleList()
                
                self.conv_layers.append(
                    GraphAttentionLayer(embed_dim, hidden_dim, dropout=dropout)
                )
                
                for _ in range(num_layers - 1):
                    self.conv_layers.append(
                        GraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout)
                    )
                
                self.output_dim = hidden_dim
            
            # Aggregation for sequence input (fallback mode)
            self.sequence_encoder = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
                bidirectional=False
            )
            
            # Output layers
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(self.output_dim, vocab_size)
            
            self._init_weights()
        
        def _init_weights(self):
            """Initialize model weights"""
            for name, param in self.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        
        def forward_graph(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            batch: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Forward pass for graph input.
            
            Args:
                x: Node type indices (num_nodes,)
                edge_index: Edge index (2, num_edges)
                batch: Batch assignment for each node
            
            Returns:
                logits: Prediction scores (batch_size, vocab_size)
            """
            # Embed node types
            h = self.embedding(x)  # (num_nodes, embed_dim)
            
            # Apply GAT layers
            for conv in self.conv_layers:
                h = conv(h, edge_index)
                h = F.elu(h)
                h = self.dropout(h)
            
            # Global pooling to get graph-level representation
            if batch is not None:
                if self.use_pyg:
                    h = global_mean_pool(h, batch)  # (batch_size, hidden_dim)
                else:
                    # Manual pooling
                    unique_batches = batch.unique()
                    pooled = []
                    for b in unique_batches:
                        mask = batch == b
                        pooled.append(h[mask].mean(dim=0))
                    h = torch.stack(pooled)
            else:
                h = h.mean(dim=0, keepdim=True)  # Single graph
            
            # Classify
            logits = self.classifier(h)
            
            return logits
        
        def forward(
            self,
            input_ids: torch.Tensor,
            edge_index: Optional[torch.Tensor] = None,
            batch: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Forward pass (compatible with existing training pipeline).
            
            If edge_index is provided, use graph mode.
            Otherwise, fall back to sequence mode for compatibility.
            """
            if edge_index is not None:
                return self.forward_graph(input_ids, edge_index, batch)
            
            # Fallback: sequence mode (for compatibility with existing code)
            embedded = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
            output, _ = self.sequence_encoder(embedded)
            output = output[:, -1, :]  # Last hidden state
            output = self.dropout(output)
            logits = self.classifier(output)
            
            return logits
        
        def predict(
            self,
            input_ids: torch.Tensor,
            edge_index: Optional[torch.Tensor] = None,
            top_k: int = 5
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Get top-k predictions"""
            self.eval()
            with torch.no_grad():
                logits = self.forward(input_ids, edge_index)
                probs = F.softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
            
            return top_indices, top_probs
        
        def get_num_params(self) -> int:
            """Count trainable parameters"""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


    class HybridNodePredictor(nn.Module):
        """
        Hybrid model combining:
        1. Node2Vec pre-trained embeddings
        2. Graph Attention Network
        3. Sequence-based LSTM (fallback)
        
        Best of both worlds for workflow prediction.
        """
        
        def __init__(
            self,
            vocab_size: int,
            embed_dim: int = 64,
            hidden_dim: int = 128,
            num_heads: int = 4,
            num_layers: int = 2,
            dropout: float = 0.2,
            pad_idx: int = 0
        ):
            super().__init__()
            
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            
            # Shared embedding
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
            
            # GAT branch
            self.gat = GraphAttentionNetwork(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                pad_idx=pad_idx
            )
            
            # LSTM branch
            self.lstm = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=dropout
            )
            
            # Fusion layer
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, vocab_size)
            )
            
            # Single-mode output
            self.classifier = nn.Linear(hidden_dim, vocab_size)
        
        def forward(
            self,
            input_ids: torch.Tensor,
            edge_index: Optional[torch.Tensor] = None,
            batch: Optional[torch.Tensor] = None,
            mode: str = "hybrid"
        ) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                input_ids: Input node indices
                edge_index: Graph edges (optional)
                batch: Batch indices (optional)
                mode: 'gat', 'lstm', or 'hybrid'
            """
            embedded = self.embedding(input_ids)
            
            if mode == "lstm" or edge_index is None:
                # LSTM only
                output, _ = self.lstm(embedded)
                output = output[:, -1, :]
                return self.classifier(output)
            
            elif mode == "gat":
                # GAT only
                return self.gat.forward_graph(input_ids.view(-1), edge_index, batch)
            
            else:
                # Hybrid: combine both
                # LSTM path
                lstm_out, _ = self.lstm(embedded)
                lstm_out = lstm_out[:, -1, :]
                
                # GAT path (if graph available)
                gat_out = self.gat.forward_graph(input_ids.view(-1), edge_index, batch)
                
                # Fuse
                combined = torch.cat([lstm_out, gat_out], dim=-1)
                return self.fusion(combined)
        
        def get_num_params(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_gat_model(
    vocab_size: int,
    architecture: str = "gat",
    **kwargs
) -> "GraphAttentionNetwork":
    """Factory function to create GAT model"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for model creation")
    
    if architecture == "gat":
        return GraphAttentionNetwork(vocab_size=vocab_size, **kwargs)
    elif architecture == "hybrid":
        return HybridNodePredictor(vocab_size=vocab_size, **kwargs)
    else:
        return GraphAttentionNetwork(vocab_size=vocab_size, **kwargs)


def workflow_to_graph_data(
    node_types: List[int],
    edges: List[Tuple[int, int]],
    target: Optional[int] = None
) -> "Data":
    """
    Convert workflow to PyTorch Geometric Data object.
    
    Args:
        node_types: List of node type indices
        edges: List of (source, target) tuples
        target: Target node for prediction (optional)
    
    Returns:
        PyG Data object
    """
    if not PYG_AVAILABLE:
        raise RuntimeError("PyTorch Geometric required for graph data")
    
    x = torch.tensor(node_types, dtype=torch.long)
    
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index)
    
    if target is not None:
        data.y = torch.tensor([target], dtype=torch.long)
    
    return data


if __name__ == "__main__":
    if TORCH_AVAILABLE:
        print("Testing GAT Model...")
        
        # Create model
        model = create_gat_model(vocab_size=100, architecture="gat")
        print(f"GAT Model created: {model.get_num_params():,} parameters")
        
        # Test sequence mode (compatibility)
        dummy_seq = torch.randint(0, 100, (4, 3))  # batch=4, seq=3
        output = model(dummy_seq)
        print(f"Sequence mode output shape: {output.shape}")
        
        # Test graph mode
        if PYG_AVAILABLE:
            dummy_nodes = torch.randint(0, 100, (10,))
            dummy_edges = torch.randint(0, 10, (2, 15))
            dummy_batch = torch.zeros(10, dtype=torch.long)
            
            output = model.forward_graph(dummy_nodes, dummy_edges, dummy_batch)
            print(f"Graph mode output shape: {output.shape}")
        
        print("✅ GAT Model tests passed!")
    else:
        print("PyTorch not available for testing")
