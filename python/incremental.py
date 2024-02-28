import networkx as nx
from typing import Any, Dict, Tuple, List

def identify_stable_ancestor_boundary_nodes(graph: nx.MultiDiGraph) -> dict:
    """
    Identifies 'stable_ancestor_boundary_nodes' in a directed acyclic graph (DAG) to optimize 
    computational efficiency by minimizing unnecessary recalculations upstream.

    A 'stable_ancestor_boundary_node' is defined as a node in the DAG where, if the values of its 
    dynamic ancestors (inputs that can change) remain unchanged, all the nodes upstream of that 
    node, including the node itself, do not require recalculation. This determination is made by 
    comparing the set of dynamic ancestors of a node to those of its successors; a 'stable_ancestor_boundary_node' 
    has a successor with at least one dynamic ancestor that the node itself does not have, marking 
    a boundary beyond which upstream nodes remain unaffected by changes in dynamic ancestors.

    Contrary to some interpretations, 'stable_ancestor_boundary_nodes' indicate that nodes upstream 
    — including the 'stable_ancestor_boundary_node' itself — do not require recalculation if their 
    dynamic ancestors have not changed. This does not imply that downstream nodes are exempt from 
    recalculations.

    :return: A dictionary mapping each 'stable_ancestor_boundary_node' to its set of dynamic ancestors, 
    indicating points in the graph up to which computations can be considered stable or unchanged given 
    no change in these ancestors.
    """

    # Initialize a dictionary to track the dynamic ancestors of each node
    dynamic_ancestors: Dict[int, set] = {node_id: set() for node_id in graph.nodes()}
    
    # Initialize the dictionary for stable ancestor boundary nodes
    stable_ancestor_boundary_nodes: Dict[str, set] = {}
    
    # Perform a topological sort of the graph to ensure we process nodes from inputs to outputs
    sorted_nodes = list(nx.topological_sort(graph))
    
    for node_id in sorted_nodes:
        node_type = graph.nodes[node_id]['node_type']
        
        # Source nodes set their dynamic ancestors based on their type
        if node_type in ['constant']:
            dynamic_ancestors[node_id] = set()
        elif node_type in ['input', 'table_array']:
            dynamic_ancestors[node_id] = {node_id}
        
        # Accumulate dynamic ancestors for function nodes from their parents
        else:
            parents = graph.predecessors(node_id)
            for parent in parents:
                dynamic_ancestors[node_id].update(dynamic_ancestors[parent])
                
            # Identify stable ancestor boundary nodes
            # These nodes mark a boundary for upstream stability
            for parent in parents:
                if len(dynamic_ancestors[parent]) < len(dynamic_ancestors[node_id]) and node_type not in ['input', 'constant', 'table_array']:
                    stable_ancestor_boundary_nodes[parent] = dynamic_ancestors[parent]
                
    return stable_ancestor_boundary_nodes

