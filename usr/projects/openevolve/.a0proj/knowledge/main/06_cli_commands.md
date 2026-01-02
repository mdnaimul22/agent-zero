# OpenEvolve CLI Commands

OpenEvolve চালানোর জন্য command reference।

## Basic Usage

```bash
# Standard run
python openevolve-run.py <initial_program> <evaluator> \
  --config <config.yaml> \
  --iterations <N>
```

## Full Command Options

```bash
python openevolve-run.py <initial_program.py> <evaluator.py> \
  --config <config.yaml>           # Configuration file path
  --iterations <N>                 # Number of iterations (overrides config)
  --checkpoint <checkpoint_dir>    # Resume from checkpoint
  --output-dir <path>              # Output directory (default: openevolve_output)
  --seed <N>                       # Random seed (overrides config)
  --log-level <LEVEL>              # DEBUG, INFO, WARNING, ERROR
```

## Common Workflows

### 1. New Evolution Run
```bash
python openevolve-run.py initial_program.py evaluator.py \
  --config config.yaml \
  --iterations 100
```

### 2. Resume from Checkpoint
```bash
python openevolve-run.py initial_program.py evaluator.py \
  --config config.yaml \
  --checkpoint openevolve_output/checkpoints/checkpoint_50/ \
  --iterations 50
```

### 3. Quick Test Run
```bash
python openevolve-run.py initial_program.py evaluator.py \
  --config config.yaml \
  --iterations 10 \
  --log-level DEBUG
```

## Visualization

```bash
# Install requirements
pip install -r scripts/requirements.txt

# Launch interactive visualizer
python scripts/visualizer.py

# Visualize specific checkpoint
python scripts/visualizer.py \
  --path openevolve_output/checkpoints/checkpoint_100/
```

### Visualizer Features
- 🌳 Evolution tree with parent-child relationships
- 📈 Performance tracking across generations
- 🔍 Code diff viewer
- 📊 MAP-Elites grid visualization
- 🎯 Multi-metric analysis

## Library Usage (No CLI)

```python
from openevolve import run_evolution, evolve_function

# Inline code evolution
result = run_evolution(
    initial_program='''
    def fibonacci(n):
        if n <= 1: return n
        return fibonacci(n-1) + fibonacci(n-2)
    ''',
    evaluator=lambda path: {"score": benchmark(path)},
    iterations=100
)

# Function evolution
def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr)-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

result = evolve_function(
    bubble_sort,
    test_cases=[([3,1,2], [1,2,3])],
    iterations=50
)
```

## Development Commands

```bash
# Install dev mode
pip install -e ".[dev]"
# or
make install

# Run tests
python -m unittest discover tests
# or
make test

# Format code
python -m black openevolve examples tests scripts
# or
make lint
```

## Docker Usage

```bash
# Pull image
docker pull ghcr.io/algorithmicsuperintelligence/openevolve:latest

# Run example
docker run --rm -v $(pwd):/app \
  ghcr.io/algorithmicsuperintelligence/openevolve:latest \
  examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --iterations 100
```

## Output Structure

```
openevolve_output/
├── checkpoints/
│   ├── checkpoint_10/
│   ├── checkpoint_20/
│   └── ...
├── logs/
│   └── evolution.log
├── best_program.py
└── best_program_info.json
```

## Environment Variables

```bash
# API Key (required)
export OPENAI_API_KEY="your-api-key"

# For Google Gemini
export OPENAI_API_KEY="your-gemini-api-key"  # Same env var

# Custom API key variable (in config)
# api_key: ${GEMINI_API_KEY}
```
