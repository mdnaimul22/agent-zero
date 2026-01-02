# System Message Best Practices

OpenEvolve এ system message সবচেয়ে important component। এটা LLM কে guide করে।

## Why System Messages Matter

- **Domain Expertise**: Problem space সম্পর্কে knowledge দেয়
- **Constraint Awareness**: কি change করা যাবে, কি যাবে না
- **Optimization Focus**: কোন দিকে improve করতে হবে
- **Error Prevention**: Common pitfalls avoid করতে help করে

## System Message Template

```yaml
prompt:
  system_message: |
    # ROLE
    You are an expert [DOMAIN] programmer.
    
    # TASK
    Your goal is to improve the code in EVOLVE-BLOCK to [SPECIFIC GOAL].
    
    # CONTEXT
    [Problem description and background]
    
    # OPTIMIZATION OPPORTUNITIES
    - [Opportunity 1]
    - [Opportunity 2]
    - [Opportunity 3]
    
    # CONSTRAINTS
    **MUST NOT CHANGE:**
    ❌ [Constraint 1]
    ❌ [Constraint 2]
    
    **ALLOWED TO OPTIMIZE:**
    ✅ [What can be changed 1]
    ✅ [What can be changed 2]
    
    # SUCCESS CRITERIA
    [How to measure success]
```

## Examples by Complexity

### Simple: General Optimization
```yaml
prompt:
  system_message: |
    You are an expert programmer specializing in optimization algorithms.
    Your task is to improve a function minimization algorithm to find 
    the global minimum reliably.
```

### Intermediate: Domain-Specific
```yaml
prompt:
  system_message: |
    You are an expert prompt engineer optimizing LLM prompts.
    
    Your improvements should:
    * Clarify vague instructions
    * Improve alignment with task outcome
    * Add examples where helpful
    * Avoid unnecessary verbosity
    
    Return only the improved prompt without explanations.
```

### Advanced: GPU Kernel Optimization
```yaml
prompt:
  system_message: |
    You are an expert Metal GPU programmer for Apple Silicon.
    
    # TARGET
    Optimize Metal Kernel for Grouped Query Attention (GQA)
    
    # OPTIMIZATION OPPORTUNITIES
    **Memory Access:**
    - Coalesced access patterns
    - Vectorized loading (SIMD)
    - Pre-compute indices
    
    **Algorithm Fusion:**
    - Combine max finding with score computation
    - Reduce data passes
    
    # CONSTRAINTS
    **MUST NOT CHANGE:**
    ❌ Kernel function signature
    ❌ Template parameter types
    ❌ Algorithm correctness
    
    **ALLOWED:**
    ✅ Memory access patterns
    ✅ Computation order
    ✅ Vectorization
```

## Iterative Creation Process

### Phase 1: Initial Draft
1. Basic system message লিখুন
2. 20-50 iterations run করুন
3. কোথায় "stuck" হচ্ছে observe করুন

### Phase 2: Refinement
4. Observed issues এর জন্য guidance add করুন
5. Domain terminology include করুন
6. Clear constraints define করুন

### Phase 3: Specialization
8. Good vs bad approaches এর examples add করুন
9. Library/framework specific guidance
10. Error avoidance patterns

### Phase 4: Meta-Optimization
12. OpenEvolve নিজেই prompt optimize করতে পারে!
13. See `examples/llm_prompt_optimization/`

## Common Pitfalls to Avoid

| ❌ Problem | ✅ Solution |
|-----------|------------|
| Too vague: "Make code better" | Specify what "better" means |
| Too restrictive | Allow useful optimizations |
| Missing context | Include domain knowledge |
| No examples | Add concrete examples |
| Ignoring artifacts | Refine based on error feedback |

## Tips

1. **Be Specific**: "Reduce memory allocations by using pre-allocated arrays"
2. **Include Domain Knowledge**: "For GPU: memory coalescing, occupancy, shared memory"
3. **Set Clear Boundaries**: What can and cannot be changed
4. **Use Emoji**: ✅ ❌ help visual clarity
5. **Iterate**: Start broad, then focus based on results
