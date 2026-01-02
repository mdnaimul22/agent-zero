# Creating an OpenEvolve Example

OpenEvolve example তৈরি করতে তিনটি essential component দরকার।

## 1. Initial Program (`initial_program.py`)

EVOLVE-BLOCK এ যে code আছে সেটাই LLM modify করবে।

### Structure
```python
# EVOLVE-BLOCK-START
def your_function():
    """
    এই function টি OpenEvolve evolve করবে
    """
    # Initial simple implementation
    pass
# EVOLVE-BLOCK-END

# Helper functions - NOT evolved
def helper_function():
    # এই code পরিবর্তন হবে না
    pass

if __name__ == "__main__":
    result = your_function()
    print(result)
```

### Critical Requirements
- ✅ **একটাই EVOLVE-BLOCK** - একাধিক নয়
- ✅ `# EVOLVE-BLOCK-START` এবং `# EVOLVE-BLOCK-END` markers
- ✅ Block এর বাইরে helper functions রাখুন
- ✅ Block এর ভিতরে শুধু evolve করতে চান সেই code

---

## 2. Evaluator (`evaluator.py`)

Evaluator প্রোগ্রাম কতটা ভালো সেটা score করে।

### Basic Structure
```python
from openevolve.evaluation_result import EvaluationResult

def evaluate(program_path: str) -> dict:
    """
    প্রোগ্রাম evaluate করে score return করে
    
    Args:
        program_path: Generated program এর path
        
    Returns:
        Dictionary with 'combined_score' (required)
    """
    try:
        # প্রোগ্রাম load করুন
        import importlib.util
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # প্রোগ্রাম run করুন
        result = program.your_function()
        
        # Score calculate করুন (higher = better)
        accuracy = calculate_accuracy(result)
        speed = calculate_speed(result)
        
        return {
            'combined_score': 0.7 * accuracy + 0.3 * speed,  # REQUIRED
            'accuracy': accuracy,
            'speed': speed,
        }
        
    except Exception as e:
        return {
            'combined_score': 0.0,  # Error হলেও return করতে হবে
            'error': str(e)
        }
```

### With Artifacts (Recommended)
```python
from openevolve.evaluation_result import EvaluationResult

def evaluate(program_path: str) -> EvaluationResult:
    try:
        # ... evaluation logic ...
        
        return EvaluationResult(
            metrics={
                'combined_score': 0.85,
                'accuracy': 0.9,
                'speed': 0.8,
            },
            artifacts={
                'debug_info': 'Useful debugging data',
                'stderr': 'Warning messages if any',
                'suggestion': 'How to improve further'
            }
        )
    except Exception as e:
        return EvaluationResult(
            metrics={'combined_score': 0.0, 'error': str(e)},
            artifacts={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'suggestion': 'Fix suggestion here'
            }
        )
```

### Critical Requirements
- ✅ **`combined_score` MUST return** - এটাই primary metric
- ✅ Higher score = better program
- ✅ Error হলেও `combined_score: 0.0` return করুন
- ✅ Artifacts use করুন debugging feedback এর জন্য

---

## 3. Configuration (`config.yaml`)

Evolution parameters control করে।

### Minimal Config
```yaml
max_iterations: 100
checkpoint_interval: 10

llm:
  api_base: "https://generativelanguage.googleapis.com/v1beta/openai/"
  models:
    - name: "gemini-2.5-flash"
      weight: 1.0
  temperature: 0.7
  max_tokens: 4000
  timeout: 120

database:
  population_size: 50
  num_islands: 3
  feature_dimensions:
    - "score"
    - "complexity"

evaluator:
  timeout: 60

prompt:
  system_message: |
    You are an expert programmer. Your goal is to improve the code
    in the EVOLVE-BLOCK to achieve better performance.
```

### Config Best Practices
- ✅ `feature_dimensions` MUST be a list, not integer
- ✅ Appropriate timeouts set করুন
- ✅ Meaningful `system_message` দিন
- ✅ Start small (50-100 iterations), then increase

---

## Common Mistakes

| ❌ Wrong | ✅ Correct |
|---------|-----------|
| `feature_dimensions: 2` | `feature_dimensions: ["score", "complexity"]` |
| Using `'total_score'` | Using `'combined_score'` |
| Multiple EVOLVE-BLOCK | Exactly one EVOLVE-BLOCK |
| Pre-computed bin indices | Raw continuous values |
