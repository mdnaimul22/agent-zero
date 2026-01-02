"""
Pattern Analysis Module
Discovers MacroNodes and suggests node replacements
"""
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import json

try:
    from ..data.parser import WorkflowGraph
    from ..data.vocabulary import CANONICAL_MAP
except ImportError:
    WorkflowGraph = None
    CANONICAL_MAP = {}


@dataclass
class MacroNode:
    """A discovered macro node pattern (frequently co-occurring nodes)"""
    pattern: Tuple[str, ...]      # Sequence of node types
    frequency: int                 # How often it appears
    workflows: List[str]          # Which workflows contain it
    suggested_name: str           # Suggested name for macro node
    
    def __repr__(self):
        return f"MacroNode({' → '.join(self.pattern)}, freq={self.frequency})"


@dataclass
class ReplacementSuggestion:
    """Suggestion to replace a node with a better alternative"""
    original_node: str
    suggested_node: str
    context_before: List[str]
    context_after: List[str]
    confidence: float
    reason: str


def extract_ngrams(
    sequence: List[str],
    n: int
) -> List[Tuple[str, ...]]:
    """Extract n-grams from a sequence"""
    if len(sequence) < n:
        return []
    return [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]


def find_macro_nodes(
    graphs: List,
    min_length: int = 2,
    max_length: int = 4,
    min_frequency: int = 5
) -> List[MacroNode]:
    """
    Find frequently co-occurring node sequences (MacroNodes)
    
    Args:
        graphs: List of WorkflowGraph objects
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        min_frequency: Minimum occurrences to be considered
    
    Returns:
        List of MacroNode patterns sorted by frequency
    """
    # Count all n-grams
    ngram_counts: Dict[Tuple[str, ...], int] = Counter()
    ngram_workflows: Dict[Tuple[str, ...], Set[str]] = defaultdict(set)
    
    for graph in graphs:
        # Get node sequence
        sequence = []
        sorted_nodes = sorted(graph.nodes, key=lambda n: (n.in_degree, n.node_id))
        
        for node in sorted_nodes:
            node_type = node.node_type
            node_type = CANONICAL_MAP.get(node_type, node_type)
            sequence.append(node_type)
        
        # Extract n-grams of various lengths
        for n in range(min_length, max_length + 1):
            for ngram in extract_ngrams(sequence, n):
                ngram_counts[ngram] += 1
                ngram_workflows[ngram].add(graph.workflow_id)
    
    # Filter by frequency and create MacroNode objects
    macro_nodes = []
    
    for pattern, freq in ngram_counts.items():
        if freq >= min_frequency:
            # Generate suggested name
            suggested_name = "_".join(p[:4].upper() for p in pattern)
            
            macro_nodes.append(MacroNode(
                pattern=pattern,
                frequency=freq,
                workflows=list(ngram_workflows[pattern]),
                suggested_name=suggested_name
            ))
    
    # Sort by frequency (descending) and length (descending)
    macro_nodes.sort(key=lambda m: (-m.frequency, -len(m.pattern)))
    
    return macro_nodes


def analyze_node_transitions(
    graphs: List,
    min_count: int = 3
) -> Dict[str, Dict[str, int]]:
    """
    Analyze what nodes commonly follow other nodes
    
    Returns:
        Dict mapping source_node -> {target_node: count}
    """
    transitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    for graph in graphs:
        # Get node sequence
        sorted_nodes = sorted(graph.nodes, key=lambda n: (n.in_degree, n.node_id))
        sequence = [CANONICAL_MAP.get(n.node_type, n.node_type) for n in sorted_nodes]
        
        # Count transitions
        for i in range(len(sequence) - 1):
            source = sequence[i]
            target = sequence[i + 1]
            transitions[source][target] += 1
    
    # Filter by minimum count
    filtered = {}
    for source, targets in transitions.items():
        filtered_targets = {t: c for t, c in targets.items() if c >= min_count}
        if filtered_targets:
            filtered[source] = filtered_targets
    
    return filtered


def suggest_replacements(
    graphs: List,
    target_workflow: "WorkflowGraph",
    transitions: Optional[Dict[str, Dict[str, int]]] = None
) -> List[ReplacementSuggestion]:
    """
    Suggest node replacements for a workflow based on global patterns
    
    Args:
        graphs: All workflow graphs (for learning patterns)
        target_workflow: The workflow to analyze
        transitions: Pre-computed transition matrix (optional)
    
    Returns:
        List of replacement suggestions
    """
    if transitions is None:
        transitions = analyze_node_transitions(graphs)
    
    suggestions = []
    
    # Get target workflow sequence
    sorted_nodes = sorted(target_workflow.nodes, key=lambda n: (n.in_degree, n.node_id))
    sequence = [CANONICAL_MAP.get(n.node_type, n.node_type) for n in sorted_nodes]
    
    # Analyze each position
    for i, current_node in enumerate(sequence):
        context_before = sequence[max(0, i-2):i]
        context_after = sequence[i+1:min(len(sequence), i+3)]
        
        # Check if there's a better node for this position
        if i > 0:
            prev_node = sequence[i - 1]
            if prev_node in transitions:
                common_successors = transitions[prev_node]
                total_count = sum(common_successors.values())
                
                # Find the most common successor
                best_successor, best_count = max(
                    common_successors.items(),
                    key=lambda x: x[1]
                )
                
                # If current node is not the most common and difference is significant
                current_count = common_successors.get(current_node, 0)
                
                if best_successor != current_node and best_count > current_count * 2:
                    confidence = best_count / total_count
                    
                    if confidence > 0.3:
                        suggestions.append(ReplacementSuggestion(
                            original_node=current_node,
                            suggested_node=best_successor,
                            context_before=context_before,
                            context_after=context_after,
                            confidence=confidence,
                            reason=f"After '{prev_node}', '{best_successor}' is used {best_count}x vs '{current_node}' {current_count}x"
                        ))
    
    # Sort by confidence
    suggestions.sort(key=lambda s: -s.confidence)
    
    return suggestions


def find_similar_workflows(
    target: "WorkflowGraph",
    graphs: List,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Find workflows most similar to the target
    
    Returns:
        List of (workflow_id, similarity_score) tuples
    """
    # Get target node types as set
    target_nodes = set(
        CANONICAL_MAP.get(n.node_type, n.node_type)
        for n in target.nodes
    )
    
    similarities = []
    
    for graph in graphs:
        if graph.workflow_id == target.workflow_id:
            continue
        
        graph_nodes = set(
            CANONICAL_MAP.get(n.node_type, n.node_type)
            for n in graph.nodes
        )
        
        # Jaccard similarity
        intersection = len(target_nodes & graph_nodes)
        union = len(target_nodes | graph_nodes)
        
        if union > 0:
            similarity = intersection / union
            similarities.append((graph.workflow_id, similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: -x[1])
    
    return similarities[:top_k]


def generate_pattern_report(
    graphs: List,
    output_path: Optional[str] = None
) -> Dict:
    """
    Generate a comprehensive pattern analysis report
    
    Returns:
        Dict containing pattern analysis results
    """
    print("Analyzing patterns...")
    
    # Find macro nodes
    macro_nodes = find_macro_nodes(graphs, min_frequency=3)
    
    # Analyze transitions
    transitions = analyze_node_transitions(graphs, min_count=2)
    
    # Compile report
    report = {
        "total_workflows": len(graphs),
        "macro_nodes": [
            {
                "pattern": list(m.pattern),
                "frequency": m.frequency,
                "suggested_name": m.suggested_name,
                "example_workflows": m.workflows[:3]
            }
            for m in macro_nodes[:50]  # Top 50
        ],
        "transition_hotspots": {
            node: sorted(targets.items(), key=lambda x: -x[1])[:5]
            for node, targets in sorted(
                transitions.items(),
                key=lambda x: -sum(x[1].values())
            )[:20]  # Top 20 source nodes
        }
    }
    
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {output_path}")
    
    return report


if __name__ == "__main__":
    print("Pattern analysis module loaded successfully")
