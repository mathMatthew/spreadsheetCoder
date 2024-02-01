"""
using the cycle_dection_graph.json which is created when loading the sc functions
in the xm_to_graph module, this script will show the graph connections focusing on a 
single function, function_name 
"""

import networkx as nx
from pyvis.network import Network
import json

# Function name to explore
function_name = "DATE"  # Replace with the function name you want to explore

# Load graph from JSON file
with open("./data/cycle_detection_graph.json", "r") as f:
    data = json.load(f)
G = nx.node_link_graph(data)

# Create a Pyvis Network instance
net = Network(
    height="800px", width="1200px", bgcolor="#222222", font_color="white", directed=True # type: ignore
)

# Check if the function name is in the graph
if function_name in G:
    # Get all successors within 2 levels
    successors = nx.single_source_shortest_path_length(G, function_name, cutoff=2)

    # Get all predecessors within 2 levels
    G_reversed = G.reverse()
    predecessors = nx.single_source_shortest_path_length(
        G_reversed, function_name, cutoff=2
    )

    # Combine successors and predecessors
    related_nodes = {node for node, level in successors.items() if level <= 2}
    related_nodes.update({node for node, level in predecessors.items() if level <= 2})

    # Filter out nodes with 1-character names
    related_nodes = {node for node in related_nodes if len(node) > 1}

    # Get related edges
    related_edges = [
        (u, v)
        for u, v in G.edges(related_nodes)
        if u in related_nodes and v in related_nodes
    ]

    # Add the related nodes and edges to the Pyvis network
    for node in related_nodes:
        if node == function_name:
            # Highlight the target node
            net.add_node(
                node,
                label=node,
                color="red",
                size=15,
                title=f"This is the {function_name} node",
            )
        else:
            net.add_node(node, label=node, size=12, title=f"This is the {node} node")

    for edge in related_edges:
        net.add_edge(edge[0], edge[1], arrows="to")
else:
    print(f"Node '{function_name}' not found in the graph.")

# Show the visualization
net.show("graph.html", notebook=False)
