# OpenEvolve Examples Gallery

OpenEvolve এর বিভিন্ন use cases এবং examples।

## Mathematical Optimization

### Function Minimization
**Path:** `examples/function_minimization/`

- **Task:** Complex non-convex function এর global minimum find করা
- **Achievement:** Random search থেকে Simulated Annealing evolve হয়েছে
- **Key Lesson:** Optimization algorithms automatic discovery

```bash
cd examples/function_minimization
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml
```

### Circle Packing
**Path:** `examples/circle_packing/`

- **Task:** Unit square এ 26 circles pack করা, sum of radii maximize করতে
- **Achievement:** AlphaEvolve paper results match করেছে (2.634/2.635)
- **Key Lesson:** Geometric heuristics থেকে mathematical optimization evolve হয়

---

## Algorithm Discovery

### Signal Processing
**Path:** `examples/signal_processing/`

- **Task:** Audio processing এর জন্য digital filters design
- **Achievement:** Novel filter designs with superior characteristics
- **Key Lesson:** Domain-specific algorithm evolution

### Rust Adaptive Sort
**Path:** `examples/rust_adaptive_sort/`

- **Task:** Data patterns এ adapt করে এমন sorting algorithm
- **Achievement:** Traditional algorithms এর beyond sorting strategies evolve হয়েছে
- **Key Lesson:** Multi-language support (Rust)

---

## Performance Optimization

### MLX Metal Kernel Optimization
**Path:** `examples/mlx_metal_kernel_opt/`

- **Task:** Apple Silicon এ attention mechanisms optimize করা
- **Achievement:** 2-3x speedup over baseline
- **Key Lesson:** Hardware-specific optimization

---

## AI & Machine Learning

### LLM Prompt Optimization
**Path:** `examples/llm_prompt_optimization/`

- **Task:** Better LLM performance এর জন্য prompts evolve করা
- **Achievement:** HotpotQA তে +23% accuracy improvement
- **Key Lesson:** Self-improving AI systems

### Symbolic Regression
**Path:** `examples/symbolic_regression/`

- **Task:** Data থেকে mathematical expressions discover করা
- **Achievement:** Scientific equations automated discovery
- **Key Lesson:** Scientific discovery automation

---

## Web & Data Processing

### Web Scraper with OptiLLM
**Path:** `examples/web_scraper_optillm/`

- **Task:** HTML pages থেকে API documentation extract করা
- **Achievement:** OptiLLM integration with test-time compute
- **Key Lesson:** LLM proxy systems integration

---

## Competitive Programming

### Online Judge Programming
**Path:** `examples/online_judge_programming/`

- **Task:** Competitive programming problems solve করা
- **Achievement:** Automated solution generation
- **Key Lesson:** External evaluation systems integration

---

## Running Any Example

```bash
# Navigate to example directory
cd examples/<example_name>

# Run evolution
python ../../openevolve-run.py initial_program.py evaluator.py \
  --config config.yaml \
  --iterations 100

# View results
python ../../scripts/visualizer.py \
  --path openevolve_output/checkpoints/checkpoint_100/
```

## Creating Your Own Example

1. **initial_program.py** - EVOLVE-BLOCK সহ starting code
2. **evaluator.py** - `combined_score` return করে
3. **config.yaml** - Evolution parameters

See `02_creating_examples.md` for detailed guide.
