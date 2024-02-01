"""This module graphs a dag.
"""
import networkx as nx
import matplotlib.pyplot as plt
from lxml import etree
from networkx.drawing.nx_pydot import graphviz_layout

# internal
import convert_xml as cxml
import xml_to_graph as x2g


graph_xml_file = r"C:\Users\matth\OneDrive\Documents\sc_v2_data\xml_outputs\DATE.xml"
graph_xml = cxml.load_3_or_5_tree(graph_xml_file)
G = x2g.build_nx_graph(graph_xml)
# Add nodes and edges to G

# Use Graphviz to layout your graph
pos = graphviz_layout(G, prog="dot")

# Draw the graph using Matplotlib
nx.draw(G, pos, with_labels=True, arrows=True)
plt.show()
