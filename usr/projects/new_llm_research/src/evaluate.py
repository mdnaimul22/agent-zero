"""
Evaluation Script for Node-Centric AI System
Evaluates trained models and generates prediction reports
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch required")
    sys.exit(1)

from src.data.parser import load_all_workflows
from src.data.vocabulary import NodeVocab
from src.data.dataset import load_samples, WorkflowDataset
from src.models.node_predictor import NodePredictor
from torch.utils.data import DataLoader


class Evaluator:
    """Model evaluation and prediction"""
    
    def __init__(
        self,
        model: NodePredictor,
        vocab: NodeVocab,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.vocab = vocab
        self.device = device
    
    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate on test set"""
        correct = {1: 0, 3: 0, 5: 0, 10: 0}
        total = 0
        
        for inputs, targets in test_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            _, top10_preds = torch.topk(outputs, k=10, dim=-1)
            
            for i, target in enumerate(targets):
                total += 1
                preds = top10_preds[i]
                
                for k in [1, 3, 5, 10]:
                    if target in preds[:k]:
                        correct[k] += 1
        
        return {
            f"top{k}_accuracy": correct[k] / total if total > 0 else 0
            for k in [1, 3, 5, 10]
        }
    
    def predict_next(self, context: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict the next node given context
        
        Args:
            context: List of node type names
            top_k: Number of predictions
        
        Returns:
            List of (node_type, probability) tuples
        """
        # Encode context
        encoded = [self.vocab.encode(nt) for nt in context]
        
        # Pad if needed
        window_size = self.model.window_size
        while len(encoded) < window_size:
            encoded.insert(0, self.vocab.pad_idx)
        
        # Take last window_size tokens
        encoded = encoded[-window_size:]
        
        # Create tensor
        input_tensor = torch.tensor([encoded], dtype=torch.long).to(self.device)
        
        # Predict
        top_indices, top_probs = self.model.predict(input_tensor, top_k=top_k)
        
        # Decode
        predictions = []
        for idx, prob in zip(top_indices[0].tolist(), top_probs[0].tolist()):
            node_type = self.vocab.decode(idx)
            predictions.append((node_type, prob))
        
        return predictions

    def generate_workflow(self, start_node="webhook", max_length=20, temperature=1.0):
        """Generate a complete workflow sequence autoregressively"""
        current_sequence = [start_node]
        
        print(f"Generating workflow starting with: {start_node}...")
        
        for _ in range(max_length):
            # Get next prediction (use Top-1 for now, or sample based on temp)
            suggestions = self.predict_next(current_sequence, top_k=5)
            
            # Simple greedy strategy: take best non-special token
            next_node = suggestions[0][0]
            
            # Avoid repeating the same node excessively (simple loop breaker)
            if len(current_sequence) >= 2 and next_node == current_sequence[-1] == current_sequence[-2]:
                # Try second best if stuck in tight loop
                next_node = suggestions[1][0]
            
            # Stop conditions
            if next_node in ["<PAD>", "<UNK>", "<END>"]:
                break
                
            current_sequence.append(next_node)
            
            # Heuristic stop: if we hit a 'response' or 'end' type node
            if "response" in next_node or "return" in next_node:
                break
        
        return current_sequence
    
    def interactive_mode(self):
        """Interactive prediction mode"""
        print("\n=== Interactive Node Prediction ===")
        print("Enter node types separated by commas (e.g., 'webhook, http_request')")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                user_input = input("Context: ").strip()
                
                if user_input.lower() in ('quit', 'exit', 'q'):
                    break
                
                if not user_input:
                    continue
                
                # Parse input
                context = [nt.strip().lower() for nt in user_input.split(",")]
                
                # Predict
                predictions = self.predict_next(context, top_k=5)
                
                print("\nPredicted next nodes:")
                for i, (node_type, prob) in enumerate(predictions, 1):
                    print(f"  {i}. {node_type}: {prob:.2%}")
                print()
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


def load_model(checkpoint_path: str, vocab: NodeVocab, device: str = "cpu") -> NodePredictor:
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model (assume LSTM architecture, can be configured)
    model = NodePredictor(
        vocab_size=vocab.size,
        embed_dim=64,
        hidden_dim=128,
        num_layers=2,
        pad_idx=vocab.pad_idx
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate Node Prediction Model")
    
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--vocab", type=str, default="data/vocabulary.json",
                        help="Path to vocabulary file")
    parser.add_argument("--samples", type=str, default="data/samples.json",
                        help="Path to test samples")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    
    # Load vocabulary
    print("Loading vocabulary...")
    vocab = NodeVocab.load(args.vocab)
    print(f"Vocabulary size: {vocab.size}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.model, vocab, args.device)
    print(f"Model loaded: {model.get_num_params():,} parameters")
    
    # Create evaluator
    evaluator = Evaluator(model, vocab, args.device)
    
    if args.interactive:
        evaluator.interactive_mode()
    else:
        # Load test samples
        print("Loading test samples...")
        samples = load_samples(args.samples)
        
        # Use last 10% as test
        test_samples = samples[int(len(samples) * 0.9):]
        
        test_dataset = WorkflowDataset(test_samples, vocab)
        test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=WorkflowDataset.collate_fn
        )
        
        # Evaluate
        print("\nEvaluating...")
        metrics = evaluator.evaluate(test_loader)
        
        print("\n=== Evaluation Results ===")
        for name, value in metrics.items():
            print(f"  {name}: {value:.2%}")
        
        # Save results
        results_path = Path(args.model).parent / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
