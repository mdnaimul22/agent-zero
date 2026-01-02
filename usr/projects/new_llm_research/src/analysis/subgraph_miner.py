"""
Advanced Subgraph Mining Module
================================
Discovers frequent subgraph patterns in workflow graphs.
Implements hierarchical MacroNode discovery for 5+ node patterns.

Inspired by gSpan (graph-based Substructure pattern mining) algorithm
but optimized for workflow DAG structures.
"""
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set, FrozenSet
from dataclasses import dataclass, field
import json
import hashlib

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from ..data.vocabulary import CANONICAL_MAP
except ImportError:
    CANONICAL_MAP = {}


@dataclass
class SubgraphPattern:
    """A discovered subgraph pattern"""
    nodes: Tuple[str, ...]           # Node types in topological order
    edges: Tuple[Tuple[int, int], ...] # Edges as (src_idx, dst_idx) within nodes
    frequency: int                    # How often pattern appears
    workflows: List[str]              # Which workflows contain it
    canonical_hash: str               # Unique identifier for the pattern
    
    def __repr__(self):
        return f"SubgraphPattern({' → '.join(self.nodes)}, edges={len(self.edges)}, freq={self.frequency})"
    
    def __hash__(self):
        return hash(self.canonical_hash)
    
    def __eq__(self, other):
        return self.canonical_hash == other.canonical_hash


@dataclass
class HierarchicalMacroNode:
    """
    MacroNode that can contain other MacroNodes.
    Represents a hierarchical abstraction of workflow patterns.
    """
    name: str
    pattern: SubgraphPattern
    children: List["HierarchicalMacroNode"] = field(default_factory=list)
    parent: Optional["HierarchicalMacroNode"] = None
    compression_ratio: float = 1.0  # How much this pattern compresses the graph
    
    def __repr__(self):
        return f"MacroNode({self.name}, children={len(self.children)}, freq={self.pattern.frequency})"
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "nodes": list(self.pattern.nodes),
            "edges": list(self.pattern.edges),
            "frequency": self.pattern.frequency,
            "compression_ratio": self.compression_ratio,
            "children": [c.to_dict() for c in self.children]
        }


class SubgraphMiner:
    """
    Advanced subgraph pattern mining for workflow graphs.
    
    Discovers:
    1. Frequent node sequences (like existing n-gram mining)
    2. Frequent subgraph patterns (with branching structure)
    3. Hierarchical MacroNodes (patterns within patterns)
    """
    
    def __init__(
        self,
        min_support: int = 3,
        max_pattern_size: int = 8,
        min_pattern_size: int = 2
    ):
        self.min_support = min_support
        self.max_pattern_size = max_pattern_size
        self.min_pattern_size = min_pattern_size
        
        self.patterns: Dict[str, SubgraphPattern] = {}
        self.macro_nodes: List[HierarchicalMacroNode] = []
    
    def _canonicalize_node_type(self, node_type: str) -> str:
        """Apply canonical mapping to node types"""
        return CANONICAL_MAP.get(node_type, node_type)
    
    def _compute_pattern_hash(
        self,
        nodes: Tuple[str, ...],
        edges: Tuple[Tuple[int, int], ...]
    ) -> str:
        """Compute a canonical hash for a subgraph pattern"""
        # Sort nodes and edges for canonical form
        pattern_str = f"nodes:{','.join(nodes)}|edges:{','.join(f'{s}-{t}' for s,t in sorted(edges))}"
        return hashlib.md5(pattern_str.encode()).hexdigest()[:16]
    
    def _extract_local_subgraphs(
        self,
        workflow_id: str,
        nodes: List[Tuple[str, str]],  # [(node_id, node_type), ...]
        edges: List[Tuple[str, str]]   # [(src_id, dst_id), ...]
    ) -> List[SubgraphPattern]:
        """
        Extract all connected subgraphs up to max_pattern_size from a workflow.
        Uses BFS from each node to enumerate connected subgraphs.
        """
        if not NETWORKX_AVAILABLE:
            return self._extract_subgraphs_simple(workflow_id, nodes, edges)
        
        # Build graph
        G = nx.DiGraph()
        node_id_to_type = {}
        
        for node_id, node_type in nodes:
            canonical_type = self._canonicalize_node_type(node_type)
            G.add_node(node_id, node_type=canonical_type)
            node_id_to_type[node_id] = canonical_type
        
        for src, dst in edges:
            if src in G and dst in G:
                G.add_edge(src, dst)
        
        patterns = []
        
        # Enumerate subgraphs starting from each node
        for start_node in G.nodes():
            subgraphs = self._enumerate_connected_subgraphs(G, start_node)
            patterns.extend(subgraphs)
        
        return patterns
    
    def _enumerate_connected_subgraphs(
        self,
        G: "nx.DiGraph",
        start_node: str
    ) -> List[SubgraphPattern]:
        """
        Enumerate all connected subgraphs containing start_node
        up to max_pattern_size nodes.
        """
        patterns = []
        
        # BFS to expand subgraphs
        queue = [(frozenset([start_node]),)]
        visited_subgraphs: Set[FrozenSet[str]] = set()
        
        while queue:
            current_nodes = queue.pop(0)
            current_set = current_nodes[0] if isinstance(current_nodes, tuple) else current_nodes
            
            if current_set in visited_subgraphs:
                continue
            visited_subgraphs.add(current_set)
            
            if len(current_set) >= self.min_pattern_size:
                # Create pattern from current subgraph
                pattern = self._create_pattern_from_nodes(G, current_set)
                if pattern:
                    patterns.append(pattern)
            
            if len(current_set) >= self.max_pattern_size:
                continue
            
            # Expand by adding neighbors
            for node in current_set:
                for neighbor in list(G.successors(node)) + list(G.predecessors(node)):
                    if neighbor not in current_set:
                        new_set = current_set | frozenset([neighbor])
                        if new_set not in visited_subgraphs:
                            queue.append((new_set,))
        
        return patterns
    
    def _create_pattern_from_nodes(
        self,
        G: "nx.DiGraph",
        node_set: FrozenSet[str]
    ) -> Optional[SubgraphPattern]:
        """Create a SubgraphPattern from a set of nodes"""
        # Get subgraph
        subgraph = G.subgraph(node_set)
        
        # Sort nodes by in-degree for canonical ordering
        sorted_nodes = sorted(node_set, key=lambda n: (G.in_degree(n), n))
        node_to_idx = {n: i for i, n in enumerate(sorted_nodes)}
        
        # Extract node types
        node_types = tuple(G.nodes[n]["node_type"] for n in sorted_nodes)
        
        # Extract edges
        edges = tuple(
            (node_to_idx[u], node_to_idx[v])
            for u, v in subgraph.edges()
        )
        
        # Compute hash
        pattern_hash = self._compute_pattern_hash(node_types, edges)
        
        return SubgraphPattern(
            nodes=node_types,
            edges=edges,
            frequency=1,
            workflows=[],
            canonical_hash=pattern_hash
        )
    
    def _extract_subgraphs_simple(
        self,
        workflow_id: str,
        nodes: List[Tuple[str, str]],
        edges: List[Tuple[str, str]]
    ) -> List[SubgraphPattern]:
        """
        Fallback subgraph extraction without NetworkX.
        Uses simpler path-based enumeration.
        """
        patterns = []
        
        # Build adjacency list
        adj = defaultdict(list)
        node_types = {}
        
        for node_id, node_type in nodes:
            node_types[node_id] = self._canonicalize_node_type(node_type)
        
        for src, dst in edges:
            adj[src].append(dst)
        
        # Enumerate paths of various lengths
        for length in range(self.min_pattern_size, self.max_pattern_size + 1):
            for start_id, _ in nodes:
                paths = self._enumerate_paths(start_id, adj, length)
                for path in paths:
                    path_types = tuple(node_types[n] for n in path)
                    path_edges = tuple((i, i+1) for i in range(len(path)-1))
                    
                    pattern_hash = self._compute_pattern_hash(path_types, path_edges)
                    
                    patterns.append(SubgraphPattern(
                        nodes=path_types,
                        edges=path_edges,
                        frequency=1,
                        workflows=[workflow_id],
                        canonical_hash=pattern_hash
                    ))
        
        return patterns
    
    def _enumerate_paths(
        self,
        start: str,
        adj: Dict[str, List[str]],
        length: int
    ) -> List[List[str]]:
        """Enumerate all paths of given length starting from start"""
        if length <= 1:
            return [[start]]
        
        paths = []
        for next_node in adj.get(start, []):
            sub_paths = self._enumerate_paths(next_node, adj, length - 1)
            for sub_path in sub_paths:
                paths.append([start] + sub_path)
        
        return paths
    
    def mine_patterns(self, graphs: List) -> List[SubgraphPattern]:
        """
        Mine frequent subgraph patterns from a list of workflow graphs.
        
        Args:
            graphs: List of WorkflowGraph objects (from parser.py)
        
        Returns:
            List of frequent SubgraphPattern objects
        """
        # Collect all patterns and count frequencies
        pattern_counts: Dict[str, SubgraphPattern] = {}
        pattern_workflows: Dict[str, Set[str]] = defaultdict(set)
        
        for graph in graphs:
            # Extract nodes and edges from graph
            nodes = [(n.node_id, n.node_type) for n in graph.nodes]
            edges = [(e.source, e.target) for e in graph.edges]
            
            local_patterns = self._extract_local_subgraphs(
                graph.workflow_id, nodes, edges
            )
            
            # Count patterns
            seen_in_this_graph = set()
            for pattern in local_patterns:
                if pattern.canonical_hash not in seen_in_this_graph:
                    seen_in_this_graph.add(pattern.canonical_hash)
                    
                    if pattern.canonical_hash in pattern_counts:
                        pattern_counts[pattern.canonical_hash].frequency += 1
                    else:
                        pattern_counts[pattern.canonical_hash] = pattern
                    
                    pattern_workflows[pattern.canonical_hash].add(graph.workflow_id)
        
        # Filter by support and update workflows
        frequent_patterns = []
        for pattern_hash, pattern in pattern_counts.items():
            if pattern.frequency >= self.min_support:
                pattern.workflows = list(pattern_workflows[pattern_hash])
                frequent_patterns.append(pattern)
        
        # Sort by frequency
        frequent_patterns.sort(key=lambda p: (-p.frequency, -len(p.nodes)))
        
        self.patterns = {p.canonical_hash: p for p in frequent_patterns}
        return frequent_patterns
    
    def build_hierarchical_macronodes(
        self,
        patterns: List[SubgraphPattern]
    ) -> List[HierarchicalMacroNode]:
        """
        Build hierarchical MacroNodes from patterns.
        A pattern is a child of another if it's fully contained within it.
        """
        # Create MacroNodes
        macro_nodes = []
        for pattern in patterns:
            name = self._generate_macro_name(pattern)
            compression = len(pattern.nodes) / 2.0  # Rough compression estimate
            
            macro_node = HierarchicalMacroNode(
                name=name,
                pattern=pattern,
                compression_ratio=compression
            )
            macro_nodes.append(macro_node)
        
        # Build parent-child relationships
        # Pattern A is child of B if A's nodes are subset of B's nodes
        for i, child in enumerate(macro_nodes):
            child_nodes = set(child.pattern.nodes)
            
            for j, parent in enumerate(macro_nodes):
                if i == j:
                    continue
                    
                parent_nodes = set(parent.pattern.nodes)
                
                if child_nodes < parent_nodes:  # Strict subset
                    parent.children.append(child)
                    if child.parent is None or len(child.parent.pattern.nodes) > len(parent.pattern.nodes):
                        child.parent = parent
        
        # Return only root nodes (those without parents)
        root_nodes = [m for m in macro_nodes if m.parent is None]
        root_nodes.sort(key=lambda m: -m.pattern.frequency)
        
        self.macro_nodes = root_nodes
        return root_nodes
    
    def _generate_macro_name(self, pattern: SubgraphPattern) -> str:
        """Generate a readable name for a MacroNode"""
        # Take first 3-4 chars of each node type
        parts = [n[:4].upper() for n in pattern.nodes[:4]]
        if len(pattern.nodes) > 4:
            parts.append(f"+{len(pattern.nodes)-4}")
        return "_".join(parts)
    
    def get_pattern_statistics(self) -> Dict:
        """Get statistics about discovered patterns"""
        if not self.patterns:
            return {"error": "No patterns mined yet"}
        
        patterns = list(self.patterns.values())
        
        # Size distribution
        size_dist = Counter(len(p.nodes) for p in patterns)
        
        # Top patterns by size
        patterns_by_size = defaultdict(list)
        for p in patterns:
            patterns_by_size[len(p.nodes)].append(p)
        
        top_by_size = {}
        for size, size_patterns in patterns_by_size.items():
            sorted_patterns = sorted(size_patterns, key=lambda x: -x.frequency)
            top_by_size[size] = [
                {"nodes": list(p.nodes), "frequency": p.frequency}
                for p in sorted_patterns[:5]
            ]
        
        return {
            "total_patterns": len(patterns),
            "size_distribution": dict(size_dist),
            "top_patterns_by_size": top_by_size,
            "max_pattern_size": max(len(p.nodes) for p in patterns),
            "avg_frequency": sum(p.frequency for p in patterns) / len(patterns)
        }
    
    def generate_report(self, output_path: Optional[str] = None) -> Dict:
        """Generate comprehensive mining report"""
        report = {
            "statistics": self.get_pattern_statistics(),
            "top_50_patterns": [
                {
                    "nodes": list(p.nodes),
                    "edges": list(p.edges),
                    "frequency": p.frequency,
                    "example_workflows": p.workflows[:3]
                }
                for p in sorted(self.patterns.values(), key=lambda x: -x.frequency)[:50]
            ],
            "hierarchical_macronodes": [
                m.to_dict() for m in self.macro_nodes[:20]
            ]
        }
        
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"Report saved to {output_path}")
        
        return report


def mine_workflow_patterns(
    graphs: List,
    min_support: int = 3,
    max_size: int = 6,
    output_path: Optional[str] = None
) -> Dict:
    """
    Convenience function to mine patterns from workflows.
    
    Args:
        graphs: List of WorkflowGraph objects
        min_support: Minimum frequency for a pattern
        max_size: Maximum pattern size to mine
        output_path: Optional path to save report
    
    Returns:
        Mining report dictionary
    """
    print(f"Mining patterns from {len(graphs)} workflows...")
    
    miner = SubgraphMiner(
        min_support=min_support,
        max_pattern_size=max_size
    )
    
    patterns = miner.mine_patterns(graphs)
    print(f"Found {len(patterns)} frequent patterns")
    
    macro_nodes = miner.build_hierarchical_macronodes(patterns)
    print(f"Built {len(macro_nodes)} hierarchical MacroNodes")
    
    return miner.generate_report(output_path)


if __name__ == "__main__":
    print("Subgraph Mining Module loaded successfully")
    print(f"NetworkX available: {NETWORKX_AVAILABLE}")
