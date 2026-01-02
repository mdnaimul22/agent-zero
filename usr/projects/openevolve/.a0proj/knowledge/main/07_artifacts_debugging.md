# Artifacts and Debugging

OpenEvolve এ artifacts system evolution improve করতে help করে।

## Artifacts কি?

Artifacts হলো side-channel data যা evaluator থেকে LLM এ feedback দেয়:
- Error messages
- Debugging info
- Performance data
- Suggestions

## Artifacts Return করা

```python
from openevolve.evaluation_result import EvaluationResult

def evaluate(program_path: str) -> EvaluationResult:
    try:
        # ... run program ...
        
        return EvaluationResult(
            metrics={
                'combined_score': 0.85,
                'accuracy': 0.9,
            },
            artifacts={
                'convergence_info': 'Converged in 50 iterations',
                'best_position': f'x={x:.4f}, y={y:.4f}',
                'performance_note': 'Good, but could optimize memory usage',
            }
        )
        
    except Exception as e:
        return EvaluationResult(
            metrics={'combined_score': 0.0},
            artifacts={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'suggestion': 'Check for division by zero',
            }
        )
```

## LLM Prompt এ কিভাবে দেখায়

```markdown
## Previous Execution Feedback
⚠️ Warning: suboptimal memory access pattern
💡 LLM Feedback: Code is correct but variable names could be better
🔧 Build Warnings: unused variable x
```

## Artifact Best Practices

### Error Artifacts
```python
artifacts = {
    'error_type': 'TimeoutError',
    'error_message': 'Function exceeded 5s timeout',
    'suggestion': 'Reduce iterations or add early termination',
}
```

### Performance Artifacts
```python
artifacts = {
    'execution_time': f'{time:.2f}s',
    'memory_peak': f'{memory_mb:.1f}MB',
    'optimization_hint': 'Consider caching repeated calculations',
}
```

### Debug Artifacts
```python
artifacts = {
    'intermediate_values': f'x={x}, y={y}',
    'algorithm_state': 'Stuck in local minimum',
    'convergence_status': 'Not converged after 1000 iterations',
}
```

## Config Settings

```yaml
prompt:
  include_artifacts: true         # Enable artifacts in prompt
  max_artifact_bytes: 20480       # 20KB max size
  artifact_security_filter: true  # Filter sensitive data
```

## Storage

- **Small artifacts** (<10KB): Stored in database
- **Large artifacts** (>10KB): Saved to disk files

## Debugging Tips

1. **Check Logs**: `openevolve_output/logs/`
2. **Failed Programs**: Checkpoint directories এ থাকে
3. **Test Evaluator Separately**: Evolution এর আগে independently test করুন
4. **Use Verbose Artifacts**: Error details সহ return করুন
