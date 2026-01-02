# OpenEvolve Overview

OpenEvolve হলো একটি evolutionary coding agent যা LLM ব্যবহার করে code automatically optimize করে।

## মূল ধারণা

OpenEvolve কোড "evolve" করে - অর্থাৎ initial code থেকে শুরু করে LLM এর মাধ্যমে iteratively improve করে optimal solution এ পৌঁছায়।

## Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                      OpenEvolve                              │
├─────────────────────────────────────────────────────────────┤
│  Initial Program → LLM Mutation → Evaluation → Selection    │
│         ↑                                         ↓         │
│         └──────────── Next Generation ←──────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### 1. Controller (`openevolve/controller.py`)
- Evolution process এর main orchestrator
- Parallel iteration execution manage করে

### 2. Database (`openevolve/database.py`)
- MAP-Elites algorithm implement করে
- Island-based evolution - multiple populations
- Programs feature grid এ map হয়

### 3. Evaluator (`openevolve/evaluator.py`)
- Cascade evaluation pattern
- Stage 1: Quick validation
- Stage 2: Full evaluation
- `combined_score` return করতে হবে

### 4. LLM Integration
- OpenAI-compatible API সব সাপোর্ট করে
- Ensemble approach - multiple models
- Gemini, OpenAI, Local models সব চলে

## Key Architecture Patterns

### EVOLVE-BLOCK Markers
```python
# EVOLVE-BLOCK-START
def function_to_evolve():
    # Only this code will be modified by OpenEvolve
    pass
# EVOLVE-BLOCK-END

# Code outside the block remains unchanged
def helper_function():
    pass
```

### Island-Based Evolution
- Multiple populations evolve separately
- Periodic migration prevents convergence
- More islands = more diversity

### Artifact System
- Side-channel for debugging data
- Error feedback improves next generation
- Small artifacts stored in DB, large saved to disk

## Running OpenEvolve

```bash
# Basic run
python openevolve-run.py initial_program.py evaluator.py \
  --config config.yaml --iterations 100

# Resume from checkpoint
python openevolve-run.py initial_program.py evaluator.py \
  --config config.yaml \
  --checkpoint checkpoint_directory \
  --iterations 50
```

## Cost Considerations

| Model | Cost per Iteration |
|-------|-------------------|
| Gemini-2.5-Flash | ~$0.01-0.05 (cheapest) |
| Gemini-2.5-Pro | ~$0.08-0.30 |
| o3-mini | ~$0.03-0.12 |
| o3 | ~$0.15-0.60 |
| Local models | Free after setup |
