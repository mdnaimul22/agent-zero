# OpenEvolve Configuration Reference

OpenEvolve এর সব configuration options এর complete reference।

## General Settings

```yaml
max_iterations: 100              # Evolution iterations সংখ্যা
checkpoint_interval: 10          # প্রতি N iteration এ checkpoint save
log_level: "INFO"                # DEBUG, INFO, WARNING, ERROR
random_seed: 42                  # Reproducibility এর জন্য seed
```

## LLM Configuration

```yaml
llm:
  # Model configuration
  models:
    - name: "gemini-2.5-flash"   # Primary model
      weight: 0.8                # Selection weight
    - name: "gemini-2.5-pro"
      weight: 0.2
  
  # API settings
  api_base: "https://generativelanguage.googleapis.com/v1beta/openai/"
  api_key: null                  # Uses OPENAI_API_KEY env var
  
  # Generation parameters
  temperature: 0.7               # Higher = more creative
  top_p: 0.95
  max_tokens: 4096
  
  # Request parameters
  timeout: 60                    # API timeout seconds
  retries: 3
  retry_delay: 5
```

### Model Options

| Provider | API Base | Model Names |
|----------|----------|-------------|
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | gemini-2.5-flash, gemini-2.5-pro |
| OpenAI | `https://api.openai.com/v1` | gpt-4, o3-mini, o3 |
| Ollama (Local) | `http://localhost:11434/v1` | codellama:7b, llama3 |
| OptiLLM | `http://localhost:8000/v1` | Any with plugins |

## Database Configuration (MAP-Elites)

```yaml
database:
  population_size: 1000          # Max programs in memory
  archive_size: 100              # Elite archive size
  num_islands: 5                 # Parallel populations
  
  # Island evolution
  migration_interval: 50         # Migration every N generations
  migration_rate: 0.1            # 10% top programs migrate
  
  # Selection
  elite_selection_ratio: 0.1     # Elite selection ratio
  exploration_ratio: 0.2
  exploitation_ratio: 0.7
  
  # Feature dimensions (MUST be list)
  feature_dimensions:
    - "complexity"               # Built-in: code length
    - "diversity"                # Built-in: structural diversity
    # Custom metrics from evaluator:
    # - "performance"
    # - "memory_usage"
  
  feature_bins: 10               # Bins per dimension
```

### Feature Dimensions Best Practices

**Return raw values, NOT bin indices:**

```python
# ✅ CORRECT - raw values
return {
    "combined_score": accuracy,
    "prompt_length": len(prompt),        # Raw count
    "execution_time": measure_runtime(), # Raw seconds
}

# ❌ WRONG - pre-computed bins
if prompt_length < 100:
    length_bin = 0  # Don't do this!
```

## Prompt Configuration

```yaml
prompt:
  system_message: |
    You are an expert programmer specializing in optimization.
    Your task is to improve the code to achieve better performance.
    
    Focus on:
    - Algorithmic improvements
    - Code optimization
    - Bug fixes
  
  num_top_programs: 3            # Top performers to show
  num_diverse_programs: 2        # Diverse examples to show
  
  # Artifact handling
  include_artifacts: true        # Include execution feedback
  max_artifact_bytes: 20480      # 20KB max artifact size
  
  # Template variations for diversity
  use_template_stochasticity: true
  template_variations:
    improvement_suggestion:
      - "Here's how we could improve this code:"
      - "I suggest the following improvements:"
```

## Evaluator Configuration

```yaml
evaluator:
  timeout: 300                   # Max evaluation time
  max_retries: 3
  
  # Cascade evaluation (filter bad programs early)
  cascade_evaluation: true
  cascade_thresholds:
    - 0.5                        # Stage 1 threshold
    - 0.75                       # Stage 2 threshold
  
  # Parallel evaluation
  parallel_evaluations: 4
  
  # LLM feedback (experimental)
  use_llm_feedback: false
  llm_feedback_weight: 0.1
```

## Evolution Settings

```yaml
# Evolution strategy
diff_based_evolution: true       # Diff-based vs full rewrites
max_code_length: 10000           # Max code characters

# Early stopping
early_stopping_patience: null    # Stop after N iterations without improvement
convergence_threshold: 0.001
early_stopping_metric: "combined_score"
```

## Example Configurations

### Quick Test
```yaml
max_iterations: 50
llm:
  models:
    - name: "gemini-2.5-flash"
      weight: 1.0
  temperature: 0.7
database:
  population_size: 25
  num_islands: 2
```

### Production Run
```yaml
max_iterations: 1000
checkpoint_interval: 50
llm:
  models:
    - name: "gemini-2.5-pro"
      weight: 0.6
    - name: "gemini-2.5-flash"
      weight: 0.4
  temperature: 0.7
database:
  population_size: 500
  num_islands: 5
  migration_interval: 20
evaluator:
  cascade_evaluation: true
  parallel_evaluations: 8
```
