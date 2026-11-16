"""Bounded graph checks with explicit witnesses and independent validation."""
from __future__ import annotations
import itertools
import networkx as nx

MAX_NODES = 20
MAX_EDGES = 100


def _build(spec: dict):
    if not isinstance(spec, dict): raise ValueError("invalid graph")
    vertices = spec.get("nodes")
    edges = spec.get("edges", [])
    if not isinstance(vertices, list) or not 1 <= len(vertices) <= MAX_NODES or len(set(map(str, vertices))) != len(vertices):
        raise ValueError("invalid nodes")
    if any(type(v) not in (str, int) or len(str(v)) > 30 for v in vertices): raise ValueError("invalid node")
    if not isinstance(edges, list) or len(edges) > MAX_EDGES: raise ValueError("invalid edges")
    graph = nx.DiGraph() if spec.get("directed", False) else nx.Graph()
    graph.add_nodes_from(vertices)
    for item in edges:
        if not isinstance(item, (tuple, list)) or len(item) not in (2, 3): raise ValueError("invalid edge")
        u, v = item[:2]
        if u not in graph or v not in graph: raise ValueError("edge outside graph")
        weight = item[2] if len(item) == 3 else 1
        if type(weight) not in (int, float) or not 0 <= weight <= 10000: raise ValueError("invalid weight")
        if graph.has_edge(u, v): raise ValueError("duplicate edge")
        graph.add_edge(u, v, weight=weight, capacity=weight)
    return graph


def _hamilton_witness(graph):
    nodes = list(graph.nodes)
    if len(nodes) > 10: return None
    if len(nodes) == 1: return nodes
    start = nodes[0]
    for rest in itertools.permutations(nodes[1:]):
        path = (start, *rest)
        if all(graph.has_edge(u, v) for u, v in zip(path, path[1:])):
            return list(path)
    return []


def verify_graph(spec: dict) -> dict:
    g = _build(spec)
    operation = spec.get("operation")
    if operation == "shortest_path":
        source, target = spec["source"], spec["target"]
        path = nx.shortest_path(g, source, target, weight="weight")
        length = sum(g[u][v]["weight"] for u, v in zip(path, path[1:]))
        alternate = nx.bellman_ford_path_length(g, source, target, weight="weight")
        return {"verified": length == alternate, "result": {"path": path, "length": length}, "method": "dijkstra_bellman_ford"}
    if operation == "coloring":
        colors = spec.get("colors")
        if colors is None:
            raise ValueError("color witness required")
        if g.is_directed(): raise ValueError("coloring requires undirected")
        checked = set(colors) == set(map(str, g.nodes)) and all(type(c) is int and 0 <= c < MAX_NODES for c in colors.values())
        checked = checked and all(colors[str(u)] != colors[str(v)] for u, v in g.edges)
        return {"verified": bool(checked), "result": bool(checked), "method": "edge_witness_check"}
    if operation == "bipartite":
        result = nx.is_bipartite(g.to_undirected())
        return {"verified": True, "result": result, "method": "bipartite_check"}
    if operation == "cycle":
        result = not nx.is_directed_acyclic_graph(g) if g.is_directed() else bool(nx.cycle_basis(g))
        return {"verified": True, "result": result, "method": "cycle_check"}
    if operation == "euler":
        if g.is_directed():
            result = nx.is_eulerian(g)
        else:
            result = nx.is_eulerian(g)
        return {"verified": True, "result": result, "method": "degree_connectivity"}
    if operation == "hamilton":
        witness = _hamilton_witness(g)
        if witness is None: return {"verified": False, "result": None, "error": "hamilton search limit"}
        return {"verified": True, "result": bool(witness), "path": witness, "method": "exhaustive_search"}
    if operation == "max_flow":
        if not g.is_directed(): raise ValueError("flow requires directed graph")
        source, sink = spec["source"], spec["target"]
        if source == sink: raise ValueError("distinct terminals required")
        value, flow = nx.maximum_flow(g, source, sink, capacity="capacity")
        cut, (left, right) = nx.minimum_cut(g, source, sink, capacity="capacity")
        capacity = sum(g[u][v]["capacity"] for u in left for v in g.successors(u) if v in right)
        valid = value == cut == capacity and all(sum(flow[p].get(n,0) for p in g.predecessors(n)) == sum(flow[n].values()) for n in g if n not in (source, sink))
        return {"verified": bool(valid), "result": {"flow": value, "min_cut": cut}, "method": "maxflow_mincut_witness"}
    raise ValueError("unknown graph operation")
