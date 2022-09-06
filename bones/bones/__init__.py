from .bones import Atomgraph
from .bones import Molecule

import networkx as nx
from networkx.readwrite import json_graph

def read_json(jg):
    # Read json graph
    g = json_graph.node_link_graph(jg)
    # Create Atomgraph from nodes and bonds
    h = Atomgraph(atom_dict=nx.get_node_attributes(g, "element"),
                  bond_list=sorted(g.edges()))

    # Copy node attributes
    nx.set_node_attributes(h, dict(g.nodes(data=True)))
    # Copy name
    h.name = g.name
    # Return Atomgraph
    return h
