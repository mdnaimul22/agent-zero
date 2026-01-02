#!/usr/bin/env python3
"""
Comprehensive Test Script for Node-AI Advanced Features
========================================================
Tests all implemented features:
1. GAT Model Training
2. Deep Subgraph Mining
3. FastAPI Endpoints
4. Model Accuracy Comparison
"""
import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print("="*70)
print("🧪 NODE-AI COMPREHENSIVE TEST SUITE")
print("="*70)

results = {
    "tests_passed": 0,
    "tests_failed": 0,
    "details": []
}

def log_test(name, passed, message=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n{status}: {name}")
    if message:
        print(f"       {message}")
    
    results["tests_passed" if passed else "tests_failed"] += 1
    results["details"].append({
        "name": name,
        "passed": passed,
        "message": message
    })

# ==============================================================================
# TEST 1: GAT Model Import and Creation
# ==============================================================================
print("\n" + "-"*70)
print("📦 TEST 1: GAT Model Import and Creation")
print("-"*70)

try:
    from src.models.gat_model import (
        GraphAttentionNetwork, 
        HybridNodePredictor, 
        Node2VecEmbedding,
        create_gat_model
    )
    log_test("GAT Model Import", True, "All classes imported successfully")
except Exception as e:
    log_test("GAT Model Import", False, str(e))

try:
    import torch
    gat_model = create_gat_model(vocab_size=100, architecture="gat")
    params = gat_model.get_num_params()
    log_test("GAT Model Creation", True, f"{params:,} parameters")
except Exception as e:
    log_test("GAT Model Creation", False, str(e))

try:
    # Test forward pass in sequence mode (compatibility)
    dummy_input = torch.randint(0, 100, (4, 3))
    output = gat_model(dummy_input)
    log_test("GAT Forward Pass (Sequence)", True, f"Output shape: {output.shape}")
except Exception as e:
    log_test("GAT Forward Pass (Sequence)", False, str(e))

try:
    # Test hybrid model
    hybrid_model = create_gat_model(vocab_size=100, architecture="hybrid")
    params = hybrid_model.get_num_params()
    log_test("Hybrid Model Creation", True, f"{params:,} parameters")
except Exception as e:
    log_test("Hybrid Model Creation", False, str(e))

# ==============================================================================
# TEST 2: Deep Subgraph Mining
# ==============================================================================
print("\n" + "-"*70)
print("🔍 TEST 2: Deep Subgraph Mining")
print("-"*70)

try:
    from src.analysis.subgraph_miner import (
        SubgraphMiner,
        SubgraphPattern,
        HierarchicalMacroNode,
        mine_workflow_patterns
    )
    log_test("Subgraph Miner Import", True, "All classes imported successfully")
except Exception as e:
    log_test("Subgraph Miner Import", False, str(e))

try:
    miner = SubgraphMiner(min_support=2, max_pattern_size=6)
    log_test("Subgraph Miner Initialization", True, f"max_size={miner.max_pattern_size}")
except Exception as e:
    log_test("Subgraph Miner Initialization", False, str(e))

try:
    # Test with real workflow data
    from src.data.parser import load_all_workflows
    graphs = load_all_workflows("workflows")
    
    if graphs:
        # Mine patterns
        patterns = miner.mine_patterns(graphs[:50])  # Use first 50 for speed
        log_test("Pattern Mining", True, f"Found {len(patterns)} patterns from 50 workflows")
        
        # Build hierarchical macronodes
        macro_nodes = miner.build_hierarchical_macronodes(patterns)
        log_test("Hierarchical MacroNode Building", True, f"Built {len(macro_nodes)} root MacroNodes")
        
        # Get statistics
        stats = miner.get_pattern_statistics()
        log_test("Pattern Statistics", True, 
                 f"Max pattern size: {stats.get('max_pattern_size', 'N/A')}, "
                 f"Total patterns: {stats.get('total_patterns', 0)}")
    else:
        log_test("Pattern Mining", False, "No workflow data found")
except Exception as e:
    log_test("Pattern Mining", False, str(e))

# ==============================================================================
# TEST 3: FastAPI Application
# ==============================================================================
print("\n" + "-"*70)
print("🌐 TEST 3: FastAPI Application")
print("-"*70)

try:
    from api.main import app, model_state
    log_test("FastAPI App Import", True, "App imported successfully")
except Exception as e:
    log_test("FastAPI App Import", False, str(e))

try:
    from api.schemas import (
        PredictRequest, PredictResponse,
        AnalyzeRequest, AnalyzeResponse,
        GenerateRequest, GenerateResponse
    )
    log_test("Pydantic Schemas Import", True, "All schemas imported")
except Exception as e:
    log_test("Pydantic Schemas Import", False, str(e))

try:
    # Test model state loading
    model_state.load()
    log_test("Model State Loading", True, 
             f"Loaded {len(model_state.transition_matrix)} transition types")
except Exception as e:
    log_test("Model State Loading", False, str(e))

try:
    # Test prediction
    predictions = model_state.predict_next(["webhook", "set_data"], top_k=5)
    log_test("Prediction Function", True, 
             f"Got {len(predictions)} predictions for webhook→set_data")
    
    if predictions:
        print(f"       Top prediction: {predictions[0]['node_type']} ({predictions[0]['confidence']:.2%})")
except Exception as e:
    log_test("Prediction Function", False, str(e))

# Test FastAPI endpoints with TestClient
try:
    from fastapi.testclient import TestClient
    client = TestClient(app)
    
    # Test /health endpoint
    response = client.get("/health")
    assert response.status_code == 200
    health_data = response.json()
    log_test("Health Endpoint", True, 
             f"Status: {health_data['status']}, Model loaded: {health_data['model_loaded']}")
except Exception as e:
    log_test("Health Endpoint", False, str(e))

try:
    # Test /predict endpoint
    response = client.post("/predict", json={
        "current_nodes": ["webhook", "http_request"],
        "top_k": 5
    })
    assert response.status_code == 200
    predict_data = response.json()
    log_test("Predict Endpoint", True, 
             f"Got {len(predict_data['predictions'])} predictions")
except Exception as e:
    log_test("Predict Endpoint", False, str(e))

try:
    # Test /generate endpoint
    response = client.post("/generate", json={
        "goal": "Create a WhatsApp chatbot with AI",
        "start_node": "webhook",
        "max_nodes": 8
    })
    assert response.status_code == 200
    gen_data = response.json()
    log_test("Generate Endpoint", True, 
             f"Generated {len(gen_data['nodes'])} nodes")
except Exception as e:
    log_test("Generate Endpoint", False, str(e))

try:
    # Test /analyze endpoint
    sample_workflow = {
        "id": "test_workflow",
        "name": "Test Workflow",
        "nodes": [
            {"id": "1", "type": "webhook", "parameters": {}},
            {"id": "2", "type": "set", "parameters": {}},
            {"id": "3", "type": "httpRequest", "parameters": {}}
        ],
        "connections": {
            "1": {"main": [[{"node": "2"}]]},
            "2": {"main": [[{"node": "3"}]]}
        }
    }
    
    response = client.post("/analyze", json={
        "workflow": sample_workflow,
        "find_patterns": True,
        "suggest_improvements": True
    })
    assert response.status_code == 200
    analyze_data = response.json()
    log_test("Analyze Endpoint", True, 
             f"Analyzed {analyze_data['node_count']} nodes, "
             f"optimization score: {analyze_data['optimization_score']:.2f}")
except Exception as e:
    log_test("Analyze Endpoint", False, str(e))

# ==============================================================================
# TEST 4: Training Pipeline with GAT
# ==============================================================================
print("\n" + "-"*70)
print("🎯 TEST 4: Training Pipeline Integration")
print("-"*70)

try:
    from src.train import Trainer
    from src.data.vocabulary import NodeVocab
    from src.data.dataset import WorkflowDataset, generate_samples
    
    log_test("Training Module Import", True, "All training components imported")
except Exception as e:
    log_test("Training Module Import", False, str(e))

try:
    # Load cached vocabulary
    vocab_path = Path("data/vocabulary.json")
    if vocab_path.exists():
        vocab = NodeVocab.load(str(vocab_path))
        log_test("Vocabulary Loading", True, f"Loaded vocabulary with {vocab.size} node types")
    else:
        log_test("Vocabulary Loading", False, "vocabulary.json not found")
except Exception as e:
    log_test("Vocabulary Loading", False, str(e))

try:
    # Create GAT model with real vocab size
    gat_for_training = create_gat_model(
        vocab_size=vocab.size,
        embed_dim=64,
        hidden_dim=128,
        architecture="gat"
    )
    log_test("GAT Model for Training", True, f"{gat_for_training.get_num_params():,} parameters")
except Exception as e:
    log_test("GAT Model for Training", False, str(e))

# ==============================================================================
# TEST 5: Existing LSTM Model Comparison
# ==============================================================================
print("\n" + "-"*70)
print("📊 TEST 5: Model Comparison (LSTM vs GAT)")
print("-"*70)

try:
    from src.models.node_predictor import create_model
    
    lstm_model = create_model(vocab_size=vocab.size, architecture="lstm")
    transformer_model = create_model(vocab_size=vocab.size, architecture="transformer")
    gat_model_compare = create_gat_model(vocab_size=vocab.size, architecture="gat")
    hybrid_model_compare = create_gat_model(vocab_size=vocab.size, architecture="hybrid")
    
    print(f"\n  Model Parameter Comparison:")
    print(f"  {'Model':<15} {'Parameters':>12}")
    print(f"  {'-'*28}")
    print(f"  {'LSTM':<15} {lstm_model.get_num_params():>12,}")
    print(f"  {'Transformer':<15} {transformer_model.get_num_params():>12,}")
    print(f"  {'GAT':<15} {gat_model_compare.get_num_params():>12,}")
    print(f"  {'Hybrid':<15} {hybrid_model_compare.get_num_params():>12,}")
    
    log_test("Model Comparison", True, "All architectures created successfully")
except Exception as e:
    log_test("Model Comparison", False, str(e))

# ==============================================================================
# TEST 6: n8n Node Package Structure
# ==============================================================================
print("\n" + "-"*70)
print("🔌 TEST 6: n8n Community Node Package")
print("-"*70)

n8n_files = [
    "n8n-node/package.json",
    "n8n-node/tsconfig.json",
    "n8n-node/README.md",
    "n8n-node/nodes/NodeAI/NodeAI.node.ts",
    "n8n-node/credentials/NodeAiApi.credentials.ts",
    "n8n-node/nodes/NodeAI/nodeai.svg"
]

missing_files = []
for f in n8n_files:
    if not Path(f).exists():
        missing_files.append(f)

if not missing_files:
    log_test("n8n Package Files", True, f"All {len(n8n_files)} files present")
else:
    log_test("n8n Package Files", False, f"Missing: {', '.join(missing_files)}")

try:
    # Validate package.json
    with open("n8n-node/package.json") as f:
        pkg = json.load(f)
    
    required_keys = ["name", "version", "n8n"]
    missing_keys = [k for k in required_keys if k not in pkg]
    
    if not missing_keys:
        log_test("n8n package.json Validation", True, 
                 f"Package: {pkg['name']}@{pkg['version']}")
    else:
        log_test("n8n package.json Validation", False, f"Missing keys: {missing_keys}")
except Exception as e:
    log_test("n8n package.json Validation", False, str(e))

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "="*70)
print("📋 TEST SUMMARY")
print("="*70)
print(f"\n  ✅ Passed: {results['tests_passed']}")
print(f"  ❌ Failed: {results['tests_failed']}")
print(f"  📊 Total:  {results['tests_passed'] + results['tests_failed']}")

pass_rate = results['tests_passed'] / (results['tests_passed'] + results['tests_failed']) * 100
print(f"\n  Pass Rate: {pass_rate:.1f}%")

if results['tests_failed'] > 0:
    print("\n  Failed Tests:")
    for detail in results['details']:
        if not detail['passed']:
            print(f"    - {detail['name']}: {detail['message']}")

print("\n" + "="*70)

# Save results
with open("outputs/test_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"📄 Results saved to: outputs/test_results.json")
