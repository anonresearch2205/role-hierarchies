import copy
from typing import List, Set, Dict

import networkx as nx
from minedgerolemining.RoleHierarchy.rh_utils import dict_to_digraph

def build_subset_dag(sets: list) -> dict:
    graph = {i: [] for i in range(len(sets))}
    for i, A in enumerate(sets):
        for j, B in enumerate(sets):
            if i == j:
                continue
            if A < B:
            # if A.issubset(B):
                graph_copy = copy.deepcopy(graph)
                G_copy = dict_to_digraph(graph_copy)
                if nx.is_directed_acyclic_graph(G_copy):
                # if i not in graph[j]:
                    graph[i].append(j)
    return graph


def all_maximal_chains(sets: List[Set]) -> List[List[int]]:
    graph = build_subset_dag(sets)
    chains_idx: List[List[int]] = []

    def dfs(path: List[int]):
        last = path[-1]
        extensions = [nxt for nxt in graph[last] if nxt not in path]

        if not extensions:
            chains_idx.append(path[:])
            return

        for nxt in extensions:
            dfs(path + [nxt])

    for start in range(len(sets)):
        dfs([start])

    seen = set()
    unique_idx_chains = []
    for chain in chains_idx:
        key = tuple(frozenset(sets[i]) for i in chain)
        if key not in seen:
            seen.add(key)
            unique_idx_chains.append(chain)
        else:
            print(f'Duplicate chain: {key}')
    return unique_idx_chains


def chains_as_sets(sets: list) -> list:
    idx_chains = all_maximal_chains(sets)
    return [[sets[i] for i in chain] for chain in idx_chains]


def pretty_print_chains(sets: list) -> list:
    # for chain in chains_as_sets(sets):
    #     parts = [str(s) for s in chain]
    #     print(" ⊂ ".join(parts))
    return chains_as_sets(sets)
