"""
Node Predictor Model
Neural network for predicting the next node in a workflow sequence
"""
import math
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available")


if TORCH_AVAILABLE:
    
    class PositionalEncoding(nn.Module):
        """Positional encoding for transformer-based models"""
        
        def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch, seq_len, d_model)
            x = x + self.pe[:x.size(1)].transpose(0, 1)
            return self.dropout(x)
    
    
    class NodePredictor(nn.Module):
        """
        Neural network for next-node prediction
        
        Supports multiple architectures:
        - 'lstm': LSTM-based sequence model
        - 'transformer': Transformer encoder-based model
        - 'simple': Simple feedforward model
        """
        
        def __init__(
            self,
            vocab_size: int,
            embed_dim: int = 64,
            hidden_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.2,
            architecture: str = "lstm",
            window_size: int = 2,
            pad_idx: int = 0
        ):
            super().__init__()
            
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.hidden_dim = hidden_dim
            self.architecture = architecture
            self.window_size = window_size
            
            # Node embedding layer
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_dim,
                padding_idx=pad_idx
            )
            
            # Architecture-specific layers
            if architecture == "lstm":
                self.encoder = nn.LSTM(
                    input_size=embed_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=False
                )
                self.output_dim = hidden_dim
                
            elif architecture == "transformer":
                self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=4,
                    dim_feedforward=hidden_dim,
                    dropout=dropout,
                    batch_first=True
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.output_dim = embed_dim
                
            elif architecture == "simple":
                self.encoder = nn.Sequential(
                    nn.Linear(embed_dim * window_size, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                self.output_dim = hidden_dim
            else:
                raise ValueError(f"Unknown architecture: {architecture}")
            
            # Output classifier
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(self.output_dim, vocab_size)
            
            # Initialize weights
            self._init_weights()
        
        def _init_weights(self):
            """Initialize model weights"""
            for name, param in self.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        
        def forward(
            self,
            input_ids: torch.Tensor,
            hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        ) -> torch.Tensor:
            """
            Forward pass
            
            Args:
                input_ids: (batch_size, seq_len) - Input node indices
                hidden: Optional hidden state for LSTM
            
            Returns:
                logits: (batch_size, vocab_size) - Prediction scores
            """
            # Embed input nodes
            embedded = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
            
            if self.architecture == "lstm":
                if hidden is None:
                    output, _ = self.encoder(embedded)
                else:
                    output, _ = self.encoder(embedded, hidden)
                # Use last hidden state
                output = output[:, -1, :]  # (batch, hidden_dim)
                
            elif self.architecture == "transformer":
                embedded = self.pos_encoding(embedded)
                output = self.encoder(embedded)
                # Use last position's output
                output = output[:, -1, :]  # (batch, embed_dim)
                
            elif self.architecture == "simple":
                # Flatten embedded sequence
                batch_size = embedded.size(0)
                output = embedded.view(batch_size, -1)  # (batch, embed_dim * seq_len)
                output = self.encoder(output)  # (batch, hidden_dim)
            
            # Apply dropout and classify
            output = self.dropout(output)
            logits = self.classifier(output)  # (batch, vocab_size)
            
            return logits
        
        def predict(
            self,
            input_ids: torch.Tensor,
            top_k: int = 5
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Get top-k predictions
            
            Args:
                input_ids: Input node indices
                top_k: Number of top predictions
            
            Returns:
                top_indices: Top-k predicted node indices
                top_probs: Corresponding probabilities
            """
            self.eval()
            with torch.no_grad():
                logits = self.forward(input_ids)
                probs = F.softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
            
            return top_indices, top_probs
        
        def get_num_params(self) -> int:
            """Count trainable parameters"""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


    class NodePredictorWithAttention(NodePredictor):
        """Extended model with attention mechanism for interpretability"""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
            # Attention layer
            self.attention = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim // 2),
                nn.Tanh(),
                nn.Linear(self.output_dim // 2, 1)
            )
        
        def forward_with_attention(
            self,
            input_ids: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass with attention weights
            
            Returns:
                logits: Prediction scores
                attention_weights: Attention over input positions
            """
            embedded = self.embedding(input_ids)
            
            if self.architecture == "lstm":
                output, _ = self.encoder(embedded)  # (batch, seq_len, hidden)
            elif self.architecture == "transformer":
                embedded = self.pos_encoding(embedded)
                output = self.encoder(embedded)
            else:
                # For simple architecture, reshape
                output = embedded
            
            # Compute attention weights
            attn_scores = self.attention(output).squeeze(-1)  # (batch, seq_len)
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            # Weighted sum
            context = torch.bmm(attn_weights.unsqueeze(1), output).squeeze(1)
            
            logits = self.classifier(self.dropout(context))
            
            return logits, attn_weights


def create_model(
    vocab_size: int,
    architecture: str = "lstm",
    **kwargs
) -> "NodePredictor":
    """Factory function to create model"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for model creation")
    
    return NodePredictor(
        vocab_size=vocab_size,
        architecture=architecture,
        **kwargs
    )


if __name__ == "__main__":
    if TORCH_AVAILABLE:
        # Test model creation
        model = create_model(vocab_size=100, architecture="lstm")
        print(f"Model created: {model.get_num_params():,} parameters")
        
        # Test forward pass
        dummy_input = torch.randint(0, 100, (4, 2))  # batch=4, seq=2
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
    else:
        print("PyTorch not available for testing")
