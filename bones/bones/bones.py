# Fixed self.node to self.nodes according to the new networkx versions (>=2.4) 

import sys
import networkx as nx
import numpy as np
import itertools as it
from networkx.readwrite import json_graph
from operator import itemgetter as ig
from pandas import Series
from collections import Counter
from mendeleev.fetch import fetch_table
from IPython import embed
    
class Atomgraph(nx.DiGraph):
    def __init__(self, atom_dict=None, bond_list=None, **kwargs):
        self._ptable = fetch_table("elements")
        
        # Init DiGraph
        super(Atomgraph, self).__init__(**kwargs)
        # If atom dict is given, add atoms as nodes
        if atom_dict is not None:
            self.add_nodes_from(sorted([(n,{"element": e})
                                        for (n,e) in atom_dict.items()]))
        # If bond list is given, add bonds
        if bond_list is not None:
            self.add_edges_from(bond_list)
    
    def __construct_series(x):
        ptable = fetch_table("elements")
        # Count occurrence of neighbors
        neighbors = x.elements
        neighbors.remove(x.root["element"])
        occurrences = Series(neighbors).value_counts()
        return occurrences
        
    def add_atoms(self, atom_dict):
        self.add_nodes_from(sorted([(n,{"element": e})
                                    for (n,e) in atom_dict.items()]))
        
    def add_bonds(self, bond_list):
        self.add_edges_from(bond_list)

    def equals(self, other):
        # Sorted edges
        edges1 = sorted(self.pairs) 
        edges2 = sorted(other.pairs)
        # Sorted nodes
        nodes1 = sorted(self.elements)
        nodes2 = sorted(other.elements)
        # Return True if edges and nodes match        
        return (edges1 == edges2) and (nodes1 == nodes2)

    def to_json(self, name=None):
        # Relabel from 0
        jg = nx.convert_node_labels_to_integers(self)
        # Set graph name
        jg.name = self.label if name is None else name
        # Return in json format
        return json_graph.node_link_data(jg)
    
    @property
    def root(self):
        n = min(self.in_degree(), key=ig(1))[0]
        return self.nodes[n]

    @property
    def root_node(self):
        return min(self.in_degree(), key=ig(1))[0]

    @property
    def elements(self):
        return nx.get_node_attributes(self, "element").values()

    @property
    def pairs(self):
        d = nx.get_node_attributes(self, "element")
        return [tuple(map(lambda x: d[x],e)) for e in self.edges()]

    @property
    def label(self):
        """Return a string label for the Atomgraph."""
        # Count element occurrences
        cs = sorted(Counter(
            [self.nodes[n]["element"] for (n,d) in self.in_degree() if d != 0]
        ).most_common())
        
        # Create label
        lbl = np.array(cs).astype(object).sum()
        
        return "{}-{}".format(self.root["element"], lbl if lbl else "")

class Molecule:
    def __init__(self, atoms=None, bonds=None, angles=None, dihedrals=None):
        self.__atom_cols = ["at_index", "atype", "NAC"]
        
        self.__atoms = atoms
        self.__bonds = bonds
        self.__angles = angles
        self.__dihedrals = dihedrals
                                
    def __eq__(self, other):
        mine = self.atoms[self.__atom_cols]
        yours = other.atoms[self.__atom_cols]

        same = mine.equals(yours) and \
               self.bonds.equals(other.bonds) and \
               self.angles.equals(other.angles) and \
               self.dihedrals.equals(other.dihedrals)

        return same
    
    def __hash__(self):
        h = 13

        h = h ^ self.__hash_single(self.atoms[self.__atom_cols])
        h = h ^ self.__hash_single(self.bonds)
        h = h ^ self.__hash_single(self.angles)
        h = h ^ self.__hash_single(self.dihedrals)

        return abs(h)

    def __hash_single(self, item):
        index = tuple(item.index)
        columns = tuple(item.columns)
        values = tuple(tuple(x) for x in item.values)
        item = tuple([index, columns, values])
        
        return hash(item)
      
    @property
    def atoms(self):
        return self.__atoms
    
    @atoms.setter
    def atoms(self, a):
        self.__atoms = a

    @property
    def bonds(self):
        return self.__bonds
    
    @bonds.setter
    def bonds(self, b):
        self.__bonds = b
        
    @property
    def angles(self):
        return self.__angles
    
    @angles.setter
    def angles(self, a):
        self.__angles = a
        
    @property
    def dihedrals(self):
        return self.__dihedrals
    
    @dihedrals.setter
    def dihedrals(self, d):
        self.__dihedrals = d

    @property
    def label(self):
        items = it.chain(*sorted(self.atoms["element"].value_counts().iteritems()))
        return "".join(map(str,items))
