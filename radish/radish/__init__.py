from .Topologizer import Topologizer

def main(args):
    """You should be able to use it like shown. """
    from radish import Topologizer

    topo = Topologizer.from_coords("coords.pdb")

    topo.topologize()

    coordinate_list = topo.extract("Ti", environment={"O": 6})
