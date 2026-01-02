"""
Training Script for Node-Centric AI System
Trains the NextNode prediction model on workflow data
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Error: PyTorch is required. Install with: pip install torch")
    sys.exit(1)

from src.data.parser import load_all_workflows, get_statistics
from src.data.vocabulary import NodeVocab
from src.data.dataset import (
    generate_samples, split_samples, 
    WorkflowDataset, save_samples, load_samples
)
from src.models.node_predictor import NodePredictor, create_model
from src.models.gat_model import GraphAttentionNetwork, create_gat_model


class Trainer:
    """Training manager for the node prediction model"""
    
    def __init__(
        self,
        model: NodePredictor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        vocab: NodeVocab,
        device: str = "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 0.01
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.device = device
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Compute loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # Accuracy tracking
        correct_top1 = 0
        correct_top3 = 0
        correct_top5 = 0
        total_samples = 0
        
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Compute accuracies
            _, top5_preds = torch.topk(outputs, k=5, dim=-1)
            
            for i, target in enumerate(targets):
                total_samples += 1
                if target == top5_preds[i, 0]:
                    correct_top1 += 1
                if target in top5_preds[i, :3]:
                    correct_top3 += 1
                if target in top5_preds[i]:
                    correct_top5 += 1
        
        avg_loss = total_loss / num_batches
        
        metrics = {
            "top1_accuracy": correct_top1 / total_samples if total_samples > 0 else 0,
            "top3_accuracy": correct_top3 / total_samples if total_samples > 0 else 0,
            "top5_accuracy": correct_top5 / total_samples if total_samples > 0 else 0,
        }
        
        return avg_loss, metrics
    
    def train(
        self,
        num_epochs: int,
        save_dir: str,
        early_stopping_patience: int = 10
    ) -> Dict:
        """
        Full training loop with early stopping
        
        Returns:
            Training history
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "metrics": []}
        
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Model: {self.model.get_num_params():,} parameters")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Track history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["metrics"].append(metrics)
            
            elapsed = time.time() - start_time
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Top-1: {metrics['top1_accuracy']:.2%} | "
                  f"Top-5: {metrics['top5_accuracy']:.2%} | "
                  f"Time: {elapsed:.1f}s")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                patience_counter = 0
                
                # Save checkpoint
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "metrics": metrics,
                    "vocab_size": self.vocab.size
                }
                torch.save(checkpoint, save_path / "best_model.pt")
                print(f"  → Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Save final training history
        with open(save_path / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Training complete! Best epoch: {self.best_epoch}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        return history


def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(description="Train Node-Centric AI Model")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, default="workflows",
                        help="Directory containing workflow JSON files")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory to save models and logs")
    parser.add_argument("--cache-dir", type=str, default="data",
                        help="Directory to cache processed data")
    
    # Model arguments
    parser.add_argument("--architecture", type=str, default="lstm",
                        choices=["lstm", "transformer", "simple", "gat", "hybrid"],
                        help="Model architecture")
    parser.add_argument("--embed-dim", type=int, default=64,
                        help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden layer dimension")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of encoder layers")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate")
    parser.add_argument("--window-size", type=int, default=3,
                        help="Context window size")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    
    # Other arguments
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cpu, cuda, mps, or auto)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--force-reprocess", action="store_true",
                        help="Force reprocessing of data")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create directories
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    cache_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Check for cached data
    vocab_path = cache_dir / "vocabulary.json"
    samples_path = cache_dir / "samples.json"
    
    if vocab_path.exists() and samples_path.exists() and not args.force_reprocess:
        print("Loading cached data...")
        vocab = NodeVocab.load(str(vocab_path))
        samples = load_samples(str(samples_path))
        print(f"Loaded vocabulary (size: {vocab.size}) and {len(samples)} samples")
    else:
        print("Processing workflow data...")
        
        # Load workflows
        graphs = load_all_workflows(args.data_dir)
        
        if not graphs:
            print("Error: No workflows found!")
            return
        
        # Print statistics
        stats = get_statistics(graphs)
        print(f"\nDataset Statistics:")
        print(f"  Workflows: {stats['total_workflows']}")
        print(f"  Total Nodes: {stats['total_nodes']}")
        print(f"  Unique Node Types: {stats['unique_node_types']}")
        
        # Build vocabulary
        vocab = NodeVocab.build(graphs, min_count=2)
        vocab.save(str(vocab_path))
        
        # Generate samples
        samples = generate_samples(graphs, vocab, window_size=args.window_size)
        save_samples(samples, str(samples_path))
        
        print(f"\nGenerated {len(samples)} training samples")
    
    # Split data
    train_samples, val_samples, test_samples = split_samples(samples, seed=args.seed)
    
    # Create datasets
    train_dataset = WorkflowDataset(train_samples, vocab, window_size=args.window_size)
    val_dataset = WorkflowDataset(val_samples, vocab, window_size=args.window_size)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=WorkflowDataset.collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=WorkflowDataset.collate_fn
    )
    
    print(f"\nData loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Create model
    if args.architecture in ["gat", "hybrid"]:
        model = create_gat_model(
            vocab_size=vocab.size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            architecture=args.architecture,
            pad_idx=vocab.pad_idx
        )
    else:
        model = create_model(
            vocab_size=vocab.size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            architecture=args.architecture,
            window_size=args.window_size,
            pad_idx=vocab.pad_idx
        )
    
    print(f"\nModel created: {model.get_num_params():,} parameters")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        device=device,
        learning_rate=args.learning_rate
    )
    
    # Train
    run_name = f"{args.architecture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_name
    
    history = trainer.train(
        num_epochs=args.epochs,
        save_dir=str(run_dir),
        early_stopping_patience=args.patience
    )
    
    # Save final config
    config = vars(args)
    config["vocab_size"] = vocab.size
    config["num_samples"] = len(samples)
    config["best_epoch"] = trainer.best_epoch
    config["best_val_loss"] = trainer.best_val_loss
    
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Training artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
