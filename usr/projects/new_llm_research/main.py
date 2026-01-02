"""
Node-Centric AI System
=======================
Main entry point for the complete system

Usage:
    python main.py parse       # Parse all workflows and generate statistics
    python main.py train       # Train the next-node prediction model
    python main.py evaluate    # Evaluate trained model
    python main.py analyze     # Run pattern analysis (MacroNodes)
    python main.py predict     # Interactive prediction mode
"""
import sys
import argparse
from pathlib import Path
from src.data.dataset import WorkflowDataset, generate_samples
from src.data.vocabulary import NodeVocab
from src.evaluate import Evaluator, load_model
from src.analysis.patterns import find_macro_nodes, generate_pattern_report

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def cmd_parse(args):
    """Parse all workflows and generate statistics"""
    from src.data.parser import load_all_workflows, get_statistics
    import json
    
    print(f"📂 Parsing workflows from: {args.data_dir}")
    
    graphs = load_all_workflows(args.data_dir)
    
    if not graphs:
        print("❌ No workflows found!")
        return
    
    stats = get_statistics(graphs)
    
    print("\n" + "="*60)
    print("📊 DATASET STATISTICS")
    print("="*60)
    print(f"  Total Workflows:        {stats['total_workflows']:,}")
    print(f"  Total Nodes:            {stats['total_nodes']:,}")
    print(f"  Total Edges:            {stats['total_edges']:,}")
    print(f"  Unique Node Types:      {stats['unique_node_types']:,}")
    print(f"  Avg Nodes/Workflow:     {stats['avg_nodes_per_workflow']:.2f}")
    print(f"  Avg Edges/Workflow:     {stats['avg_edges_per_workflow']:.2f}")
    
    print("\n📈 Top 20 Node Types:")
    for i, (nt, count) in enumerate(stats['top_20_node_types'], 1):
        print(f"  {i:2d}. {nt:30s} {count:5,}")
    
    # Save statistics
    output_path = Path(args.output_dir) / "dataset_statistics.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Statistics saved to: {output_path}")


def cmd_train(args):
    """Train the model"""
    import subprocess
    
    cmd = [
        sys.executable, "-m", "src.train",
        "--data-dir", args.data_dir,
        "--output-dir", args.output_dir,
        "--epochs", str(args.epochs),
        "--architecture", args.architecture
    ]
    
    if args.device:
        cmd.extend(["--device", args.device])
    
    print(f"🚀 Starting training...")
    subprocess.run(cmd)


def cmd_evaluate(args):
    """Evaluate trained model"""
    import subprocess
    
    model_path = args.model or str(Path(args.output_dir) / "lstm_latest" / "best_model.pt")
    
    cmd = [
        sys.executable, "-m", "src.evaluate",
        "--model", model_path,
        "--vocab", str(Path(args.cache_dir) / "vocabulary.json")
    ]
    
    if args.interactive:
        cmd.append("--interactive")
    
    subprocess.run(cmd)


def cmd_analyze(args):
    """Run pattern analysis"""
    from src.data.parser import load_all_workflows
    from src.analysis.patterns import find_macro_nodes, generate_pattern_report
    
    print(f"🔍 Analyzing patterns in: {args.data_dir}")
    
    graphs = load_all_workflows(args.data_dir)
    
    if not graphs:
        print("❌ No workflows found!")
        return
    
    # Generate pattern report
    output_path = Path(args.output_dir) / "pattern_analysis.json"
    output_path.parent.mkdir(exist_ok=True)
    
    report = generate_pattern_report(graphs, str(output_path))
    
    print("\n" + "="*60)
    print("🧩 DISCOVERED MACRONODES")
    print("="*60)
    
    for i, macro in enumerate(report["macro_nodes"][:15], 1):
        pattern_str = " → ".join(macro["pattern"])
        print(f"  {i:2d}. [{macro['frequency']:3d}x] {pattern_str}")
    
    print(f"\n✅ Full report saved to: {output_path}")


def cmd_predict(args):
    """Interactive prediction mode"""
    # Find the latest model
    from pathlib import Path
    output_dir = Path(args.output_dir)
    model_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("lstm")], 
                       key=lambda x: x.name, reverse=True)
    
    if model_dirs:
        model_path = str(model_dirs[0] / "best_model.pt")
    else:
        model_path = args.model
    
    # Load vocab and model
    vocab_path = Path(args.cache_dir) / "vocabulary.json"
    if not vocab_path.exists():
        print(f"❌ Vocabulary not found at {vocab_path}")
        return
        
    vocab = NodeVocab.load(str(vocab_path))
    model = load_model(model_path, vocab, device=args.device)
    
    evaluator = Evaluator(
        model=model,
        vocab=vocab,
        device=args.device
    )
    
    print("\n🔮 Interactive Prediction Mode (Type 'exit' to quit)")
    print("Enter a sequence of nodes (comma-separated):")
    
    while True:
        try:
            user_input = input("> ").strip().lower()
            if user_input in ['exit', 'quit']:
                break
                
            sequence = [s.strip() for s in user_input.split(',')]
            suggestions = evaluator.predict_next(sequence)
            
            print(f"\nNext likely nodes for sequence {sequence}:")
            for i, (node, prob) in enumerate(suggestions):
                print(f"  {i+1}. {node} ({prob*100:.1f}%)")
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

def cmd_generate(args):
    """Generate complete workflows"""
    # Find the latest model
    from pathlib import Path
    output_dir = Path(args.output_dir)
    model_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("lstm")], 
                       key=lambda x: x.name, reverse=True)
    
    if model_dirs:
        model_path = str(model_dirs[0] / "best_model.pt")
    else:
        model_path = args.model
        
    # Load vocab and model
    vocab_path = Path(args.cache_dir) / "vocabulary.json"
    if not vocab_path.exists():
        print(f"❌ Vocabulary not found at {vocab_path}")
        return

    vocab = NodeVocab.load(str(vocab_path))
    model = load_model(model_path, vocab, device=args.device)
    
    evaluator = Evaluator(model=model, vocab=vocab, device=args.device)
    
    print(f"\n🚀 Generating {args.count} workflows starting with '{args.start_node}'...\n")
    
    for i in range(args.count):
        sequence = evaluator.generate_workflow(start_node=args.start_node, max_length=args.length)
        
        print(f"Workflow #{i+1}:")
        # Print flow with arrows
        flow = " -> ".join([f"[{node}]" for node in sequence])
        print(f"{flow}")
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="🧠 Node-Centric AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py parse                    # Parse workflows and show statistics
  python main.py train --epochs 30        # Train model for 30 epochs
  python main.py analyze                  # Find MacroNode patterns
  python main.py predict                  # Interactive prediction mode
        """
    )
    
    # Global arguments
    parser.add_argument("--data-dir", type=str, default="workflows",
                        help="Directory containing workflow JSON files")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory for outputs")
    parser.add_argument("--cache-dir", type=str, default="data",
                        help="Directory for cached data")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Parse command
    parse_parser = subparsers.add_parser("parse", help="Parse workflows")
    parse_parser.set_defaults(func=cmd_parse)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--architecture", type=str, default="lstm",
                               choices=["lstm", "transformer", "simple", "gat", "hybrid"])
    train_parser.add_argument("--device", type=str, default=None)
    train_parser.set_defaults(func=cmd_train)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    eval_parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    eval_parser.add_argument("--cache_dir", type=str, default="data", help="Cache directory")
    eval_parser.add_argument("--device", type=str, default="cpu")
    eval_parser.add_argument("--interactive", action="store_true")
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Pattern analysis")
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Predict command
    parser_predict = subparsers.add_parser("predict", help="Interactive prediction")
    parser_predict.add_argument("--model", type=str, default="outputs/best_model.pt", help="Path to model checkpoint")
    parser_predict.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser_predict.add_argument("--cache_dir", type=str, default="data", help="Cache directory")
    parser_predict.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser_predict.set_defaults(func=cmd_predict)
    
    # Generate command
    parser_generate = subparsers.add_parser("generate", help="Generate complete workflows")
    parser_generate.add_argument("--start_node", type=str, default="webhook", help="Start node type")
    parser_generate.add_argument("--length", type=int, default=10, help="Max length")
    parser_generate.add_argument("--count", type=int, default=3, help="Number of workflows to generate")
    parser_generate.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser_generate.add_argument("--cache_dir", type=str, default="data", help="Cache directory")
    parser_generate.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser_generate.set_defaults(func=cmd_generate)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Route to appropriate command
    commands = {
        "parse": cmd_parse,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "analyze": cmd_analyze,
        "predict": cmd_predict,
        "generate": cmd_generate
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()
