import numpy as np
import pandas as pd
import mdtraj as md
import networkx as nx
from pandas import Series,DataFrame
from mendeleev.fetch import fetch_table
from tqdm import tqdm
from IPython import embed
from ctopo import compute_neighborlist_rad

ptable = fetch_table("elements")

class Topologizer:
    """Object which contains the molecular topology of a trajectory. """
    
    def __init__(self, bondfactor=0.25):
        """Init Object.
        
        :param bond_factor: Factor between covalent and vdw radii 
        (between 0 = covalent and 1 = vdw) 
        :returns: 
        :rtype: 
        
        """
        self.bondgraph = []
        self.kinds = []
        self.top = None
        self.trj = None
        self._bondfactor = bondfactor

    @classmethod
    def from_coords(cls, trjfile, **kwargs):
        """ Create Topologizer based on coordinates"""
        if "top" in kwargs:
            if kwargs["top"] is None:
                del kwargs["top"]
        # Bond factor
        bondfactor = kwargs.pop("bondfactor", 0.25)
        
        # Load trajectory
        trj = md.load(trjfile, **kwargs)
        # Topology DataFrame
        top,_ = trj.top.to_dataframe()

        topo = cls(bondfactor)
        topo.trj = trj
        topo.top = top

        return topo
    
    @classmethod
    def from_mdtraj(cls, trj, **kwargs):
        """ Create Topologizer based on mdtraj object"""
        # Bond factor
        bondfactor = kwargs.pop("bondfactor", 0.25)

        # Topology DataFrame
        top,_ = trj.top.to_dataframe()

        topo = cls(bondfactor)
        topo.trj = trj
        topo.top = top

        return topo
          
    def topologize(self, silent=False):
        """Construct the bondgraph for the trajectory"""
        ptab = ptable[ptable.symbol.isin(self.top.element.unique())]
        rcov = ptab.set_index("atomic_number").covalent_radius_cordero/1000
        rvdw = ptab.set_index("atomic_number").vdw_radius_alvarez/1000

        # Use bones criterium to setup dict of cutoffs
        cutoffs = (rcov + self._bondfactor*(rvdw - rcov)).to_dict()
        
        bondgraphs = []
        for fr in tqdm(self.trj, total=self.trj.n_frames, leave=False, disable=silent):
            # Compute RAD neighborlist with Cython
            nblist = compute_neighborlist_rad(fr, cutoffs)
            # Store atoms with neighbors in the bondgraph
            bondgraphs.append({k: v for (k,v) in nblist.items() if v})

        # Put neighborlists together into bondgraph DataFrame
        df = pd.concat([
            DataFrame(np.column_stack((
                i*np.ones(len(np.repeat(np.fromiter(bg.keys(), dtype=int),
                                        np.fromiter(map(len,bg.values()), dtype=int)))),
                np.repeat(np.fromiter(bg.keys(), dtype=int),
                          np.fromiter(map(len,bg.values()), dtype=int)),
                np.vstack((bg.values()))
            )), columns=["frame","i","j","dist"]
            )
            for (i,bg) in enumerate(bondgraphs)], ignore_index=True)
            
        # Correct data types
        dtype = {"frame": np.int, "i": np.int, "j": np.int, "dist": np.float}
        for (k,v) in dtype.items():
            df[k] = df[k].astype(v)

        # Store bondgraph
        self.bondgraph = df

        # Add elements of the j's
        df = df.assign(ej=lambda x: x.j.map(self.top.element))

        # Group by frame and atom, count elements, sort and unstack
        df = (df
              .groupby(["frame", "i"]).ej
              .value_counts()
              .sort_index(level=0)
              .unstack()
        )

        # Add atoms with no neighbors ("lone wolfs")
        lone_wolf_list = [
            DataFrame(columns=df.columns,
                      index=pd.MultiIndex.from_product(([], np.array([]))))
        ]
        for (n,g) in df.groupby("frame"):
            isolates = np.setdiff1d(np.arange(self.trj.n_atoms),
                                    g.index.get_level_values("i"))
            isolates_as_index = pd.MultiIndex.from_product(([n], isolates))
            lone_wolfs = DataFrame(columns=df.columns, index=isolates_as_index)
            lone_wolf_list.append(lone_wolfs)
            
        df = (df
              .combine_first((pd
                              .concat(lone_wolf_list)
                              .rename_axis(["frame","i"])))
              .reset_index()
              .fillna(0)
              .apply(pd.to_numeric, errors="ignore", downcast="integer")
              .set_index(["frame","i"])
        )

        # Construct "kind" labels
        labels = np.resize(df.columns, df.shape) + df.values.astype("str")
        labels[df==0] = ""
        labels = labels.sum(axis=1)

        # DataFrame with elements and their kinds
        df = (df.index
              .to_frame()
              .pipe(lambda x: pd.concat({"element": x.i.map(self.top.element),
                                         "kind": Series(labels, index=x.index)},
                                        axis=1))
              .reset_index()
        )

        # Store kinds
        self.kinds = df
        
    def topologize_iter(self, silent=True):
        """Construct the bondgraph of the trajectory as an iterator for each frame. """
        for fr in tqdm(self.trj, total=self.trj.n_frames, leave=False, disable=silent):
            topo = Topologizer.from_mdtraj(fr)
            topo.topologize(silent=silent)
            yield topo

    def extract(self, element, environment=None, strict=False):
        """Extract atomic environments."""
        
        # Grab elements in bondgraph
        elements = self.bondgraph[self.bondgraph["i"].map(self.top.element)==element]

        # If elements has no bonds
        if elements.empty:
            # Grab atoms directly
            atoms = self.trj.top.select("element == '{}'".format(element))
            # And coordinates
            xyz = self.trj.xyz[:,atoms,:]
            # Set up DataFrame
            data = pd.concat(
                [DataFrame(x[0],
                           columns=["x","y","z"],
                           index=pd.MultiIndex.from_arrays((np.full(x.shape[1],i),atoms),
                                                           names=("frame","i")))
                 for (i,x) in enumerate(np.split(xyz, self.trj.n_frames))])
            # Kind is empty
            data = data.assign(kind="")
            # Return data
            return data
            
        elements = elements.assign(env=elements["j"].map(self.top.element))
        # Calculate environments
        env = elements.groupby(["frame","i"]).env\
                      .value_counts()\
                      .unstack()\
                      .fillna(0)
        
        if environment is None:
            atoms_in_environment = env.index.values
        else:
            # Setup target environment
            target_env_template = np.array(list(environment.items())).T
            
            target_env = DataFrame(np.tile(target_env_template[-1],
                                           (len(env), 1)).astype(int),
                                   columns=target_env_template[0],
                                   index=env.index)
            target_env = target_env.reindex(columns=env.columns).fillna(0)
            # Match
            if strict:
                # Strict matching
                atoms_in_environment = env[(env==target_env).all(axis=1)]
                atoms_in_environment = atoms_in_environment.index.values
            else:
                # Accept if there are other atoms besides target in environment
                atoms_in_environment = env[((env==target_env)&(env!=0)).any(axis=1)]
                atoms_in_environment = atoms_in_environment.index.values

        # If no environments match
        if atoms_in_environment.size == 0:
            return DataFrame()
        
        # Compute kind labels
        env_counts = env.loc(axis=0)[atoms_in_environment]
        env_counts = env_counts.astype(int)[sorted(env.columns)].values
        env_elements = np.tile(sorted(env.columns), (len(atoms_in_environment),1))
        env_elements_and_counts = env_elements + env_counts.astype(str)\
                                                           .astype("object")
        env_elements_and_counts[env_counts==0] = ""
        kind_labels = env_elements_and_counts.sum(axis=1)

        # Extract coordinates from trajectory
        extract_indices = np.ravel_multi_index(
            np.array(atoms_in_environment.tolist()).T.tolist(),
            (self.trj.n_frames,self.trj.n_atoms)
        )
        
        extract_mask = np.zeros((self.trj.n_frames,self.trj.n_atoms))
        np.put(extract_mask,extract_indices, 1)
        extract_mask = np.repeat(np.expand_dims(extract_mask,axis=-1), 3, axis=-1)\
                         .astype(bool)
        # Collect data
        data = DataFrame(
            self.trj.xyz[extract_mask].reshape(-1,3),
            columns=["x","y","z"],
            index=pd.MultiIndex.from_tuples(atoms_in_environment.tolist()))
        data = data.assign(kind=kind_labels).rename_axis(["frame","i"]).reset_index()

        return data

    def atom_kinds(self, frames=None, atoms=None):
        """Return kinds for *frames* and *atoms*"""
        # All frames and atoms if None
        if frames is None:
            frames = np.sort(self.kinds.frame.unique())
        if atoms is None:
            atoms = np.sort(self.kinds.i.unique())

        # Assure lists
        frame_list = np.atleast_1d(frames).reshape(-1)
        atom_list = np.atleast_1d(atoms).reshape(-1)

        # The condition for returns
        cond = self.kinds.frame.isin(frame_list) & self.kinds.i.isin(atom_list)

        # Return entries from kindlist that fulfills this condition
        return self.kinds[cond]

    def neighbors(self, atoms, frames=None):
        """Return neighbors to *atoms* in *frames*"""
        # All frames if None
        if frames is None:
            frames = np.sort(self.kinds.frame.unique())

        # Assure lists
        frame_list = np.atleast_1d(frames).reshape(-1)
        atom_list = np.atleast_1d(atoms).reshape(-1)
        
        # The condition for returns
        cond = self.bondgraph.frame.isin(frame_list) & self.bondgraph.i.isin(atom_list)

        # Extract the part of the bondgraph that fulfills the condition
        df = self.bondgraph[cond]
        
        # Add atomic kinds
        kinds = self.atom_kinds(atoms=df.j).set_index(["frame","i"])
        kinds = kinds.loc[df.set_index(["frame","j"]).index]
        kinds = kinds.set_index(df.index)

        df = pd.concat([df, kinds], axis=1)

        # Arrange column order
        df = df.filter(["frame", "i", "j", "element", "kind", "dist"])

        return df

    def paths(self, N, include_shorter=False):
        """Return all N-paths in bondgraph
        
        An N-path is a path such that j can be reached from i in N steps, i.e.,
        start- and end points included.
        
        Parameters
        ----------
        N: int
            Length of path
        include_shorter: bool
            Return paths with len < N as well.

        Returns
        -------
        out : Series
            Multiindex Series with index levels [frame, i, j] and values are 
            lists with the N-paths.

        """

        # Loop over frames and find paths
        paths = {}
        for (n,bond_graph) in self.bondgraph.groupby("frame"):
            # Bond graph as network
            G = nx.from_pandas_edgelist(bond_graph,
                                        "i",
                                        "j",
                                        edge_attr="dist")
            G = nx.relabel_nodes(G, mapping=dict(zip(G.nodes(),
                                                     map(int, G.nodes()))))
            G.add_nodes_from(np.arange(self.trj.n_atoms))
            # Find shortest paths between all pairs
            frame_paths = pd.concat(
                {k: Series(v) for (k, v) in nx.all_pairs_shortest_path(G, N-1)}
            )
            paths[n] = frame_paths
        # Concatenate as Series
        paths = pd.concat(paths, names=["frame", "i", "j"])
        
        # Prune paths shorter than N
        if not include_shorter:
            paths = paths.loc[paths.agg(len)==N]

        return paths
