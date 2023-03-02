
from collections import deque
from typing import (Any, Callable, cast, Collection, Dict, FrozenSet, Iterator, List,
                    Literal, Mapping, Optional, overload, Sequence, Set, Tuple, Union)
from pauliopt.topologies import Topology

def prims_algorithm_weight(nodes: Collection[int], weight: Callable[[int, int], int],
                            inf: int) -> int:
    """
        A modified version of Prim's algorithm that
        computes the weight of the minimum spanning tree connecting
        the given nodes, using the given weight function for branches.
        The number `inf` should be larger than the maximum weight
        that can be encountered in the process.
    """
    if not nodes:
        return 0
    mst_length: int = 0
    # Initialise set of nodes to visit:
    to_visit = set(nodes)
    n0 = next(iter(to_visit))
    to_visit.remove(n0)
    # Initialise dict of distances from visited set:
    dist_from_visited: Dict[int, int] = {
        n: weight(n0, n) for n in nodes
    }
    while to_visit:
        # Look for the node to be visited which is nearest to the visited set:
        nearest_node = 0 # dummy value
        nearest_dist: int = inf # dummy value
        for n in to_visit:
            n_dist: int = dist_from_visited[n]
            if n_dist < nearest_dist:
                nearest_node = n
                nearest_dist = n_dist
        # Nearest node is removed and added to the MST:
        to_visit.remove(nearest_node)
        mst_length += nearest_dist
        # Update shortest distances to visited set:
        for n in to_visit:
            dist_nearest_n = weight(nearest_node, n)
            if dist_nearest_n < dist_from_visited[n]:
                dist_from_visited[n] = dist_nearest_n
    return mst_length

def prims_algorithm_branches(nodes: Collection[int], weight: Callable[[int, int], int],
                              inf: int) -> Sequence[Tuple[int, int]]:
    # pylint: disable = too-many-locals
    if not nodes:
        return []
    mst_branches = []
    # Initialise set of nodes to visit:
    to_visit = set(nodes)
    n0 = next(iter(to_visit))
    to_visit.remove(n0)
    # Initialise dict of distances from visited set:
    dist_from_visited: Dict[int, int] = {
        n: weight(n0, n) for n in nodes
    }
    # Initialise possible edges for the MST:
    edge_from_visited: Dict[int, Tuple[int, int]] = {
        n: (n0, n) for n in nodes
    }
    while to_visit:
        # Look for the node to be visited which is nearest to the visited set:
        nearest_node = 0 # dummy value
        nearest_dist: int = inf # dummy value
        for n in to_visit:
            n_dist: int = dist_from_visited[n]
            if n_dist < nearest_dist:
                nearest_node = n
                nearest_dist = n_dist
        # Nearest node is removed and added to the MST:
        to_visit.remove(nearest_node)
        mst_branches.append(edge_from_visited[nearest_node])
        # Update shortest distances/edges to visited set:
        for n in to_visit:
            dist_nearest_n = weight(nearest_node, n)
            if dist_nearest_n < dist_from_visited[n]:
                dist_from_visited[n] = dist_nearest_n
                edge_from_visited[n] = (nearest_node, n)
    return mst_branches

def prims_algorithm_full(nodes: Collection[int], weight: Callable[[int, int], int],
                          inf: int) -> Tuple[int, Sequence[int], Sequence[Tuple[int, int]]]:
    # pylint: disable = too-many-locals
    if not nodes:
        return 0, [], []
    mst_length: int = 0
    mst_branch_lengths = []
    mst_branches = []
    # Initialise set of nodes to visit:
    to_visit = set(nodes)
    n0 = next(iter(to_visit))
    to_visit.remove(n0)
    # Initialise dict of distances from visited set:
    dist_from_visited: Dict[int, int] = {
        n: weight(n0, n) for n in nodes
    }
    # Initialise possible edges for the MST:
    edge_from_visited: Dict[int, Tuple[int, int]] = {
        n: (n0, n) for n in nodes
    }
    while to_visit:
        # Look for the node to be visited which is nearest to the visited set:
        nearest_node = 0 # dummy value
        nearest_dist: int = inf # dummy value
        for n in to_visit:
            n_dist: int = dist_from_visited[n]
            if n_dist < nearest_dist:
                nearest_node = n
                nearest_dist = n_dist
        # Nearest node is removed and added to the MST:
        to_visit.remove(nearest_node)
        mst_length += nearest_dist
        mst_branch_lengths.append(mst_length)
        mst_branches.append(edge_from_visited[nearest_node])
        # Update shortest distances/edges to visited set:
        for n in to_visit:
            dist_nearest_n = weight(nearest_node, n)
            if dist_nearest_n < dist_from_visited[n]:
                dist_from_visited[n] = dist_nearest_n
                edge_from_visited[n] = (nearest_node, n)
    return mst_length, mst_branch_lengths, mst_branches


def topDownMSTTraversal(tree:Sequence[Tuple[int, int]], root:int, vertices:Sequence[int]) -> Sequence[Tuple[int,int]]:
    edges = []
    incident: Dict[int, Set[Tuple[int, int]]] = {q: set() for q in vertices}
    for fst, snd in tree:
        incident[fst].add((fst, snd))
        incident[snd].add((snd, fst))
    visited: Set[int] = set()
    queue = deque([root])
    while queue:
        q = queue.popleft()
        visited.add(q)
        for head, tail in incident[q]:
            if tail not in visited:
                edges.append((head, tail))
                queue.append(tail)
    return edges

def bottomUpMSTTraversal(tree:Sequence[Tuple[int, int]], root:int, vertices:Sequence[int]) -> Sequence[Tuple[int,int]]:
    edges = [(t, h) for h,t in topDownMSTTraversal(tree, root, vertices)]
    edges.reverse()
    return edges
    edges = []
    incident: Dict[int, Set[Tuple[int, int]]] = {q: set() for q in vertices}
    for fst, snd in tree:
        incident[fst].add((fst, snd))
        incident[snd].add((snd, fst))
    visited: Set[int] = set()
    queue = deque([v for v in vertices if v != root])
    while queue:
        q = queue.popleft()
        visited.add(q)
        for head, tail in incident[q]: # Edges are traversed in reverse direction wrt _preOrderTraversalEdges
            if tail not in visited:
                edges.append((head, tail))
                if tail != root:
                    queue.append(tail)
                else: 
                    visited.add(tail)
    return edges

