from itertools import combinations, permutations

import numpy as np

import gurobipy as gp
from gurobipy import GRB, quicksum
from time import time


def kendalltau_dist(ranking_a, ranking_b):
    tau = 0
    n_candidates = len(ranking_a)
    for i, j in combinations(range(n_candidates), 2):
        tau += (np.sign(ranking_a[i] - ranking_a[j]) ==
                -np.sign(ranking_b[i] - ranking_b[j]))
    return tau


def kemeny_dist(rankings, candidate_ranking):
    kemeny_dist = np.sum(kendalltau_dist(candidate_ranking, ranking) for ranking in rankings)
    return kemeny_dist


def rankaggr_brute(ranks):
    min_dist = np.inf
    best_rank = None
    n_voters, n_candidates = ranks.shape
    i=0
    for candidate_rank in permutations(range(n_candidates)):
        candidate_rank=np.argsort(candidate_rank)
        print()
        dist = np.sum(kendalltau_dist(candidate_rank, rank) for rank in ranks)
        if dist < min_dist:
            min_dist = dist
            best_rank = candidate_rank
    return min_dist, best_rank

def aggregate_kemeny(n_voters, n_candidates, ranks):

    #Declare gurobi model object
    m =gp.Model("aggregate_kemeny")
    m.setParam("OutputFlag", 0)
    print(1)

    # Indicator variable for each pair
    x = {}
    c=0
    for i in range(n_candidates):
        for j in range(n_candidates):
            x[c] = m.addVar(vtype=GRB.BINARY, name="x(%d)(%d)" %(i,j))
            c+=1
    m.update()

    idx = lambda i, j: n_candidates * i + j

    # pairwise constraints
    for i, j in combinations(range(n_candidates), 2):
        m.addConstr(x[idx(i, j)] + x[idx(j, i)] == 1)
    m.update()
    print(2)
    #transitivity constraints
    for i, j, k in permutations(range(n_candidates), 3):
        m.addConstr(x[idx(i, j)] + x[idx(j, k)] + x[idx(k,i)] >= 1)
    m.update()
    print(3)
    # Set objective
    # maximize c.T * x
    edge_weights = build_graph(ranks)
    c = -1 * edge_weights.ravel()
    m.setObjective(quicksum(c[i]*x[i] for i in range(len(x))), GRB.MAXIMIZE)
    m.update()
    #m.write("kemeny_n"+str(n_ranks)+"_N"+str(rank_len) + "_t"+str(theta) + ".lp")
    print(4)
    t0 = time()
    m.optimize()
    t1 = time()
    print(5)
    if m.status == GRB.OPTIMAL:
        #m.write("kemeny_n"+str(rank_len)+"_N"+str(n_ranks)+"_t"+str(theta)+".sol")

        #get consensus ranking
        sol = []
        for i in x:
            sol.append(x[i].X)
        sol=np.array(sol)
        aggr_rank = np.sum(sol.reshape((n_candidates,n_candidates)), axis=1)
        return aggr_rank,t1-t0
    else:
        return None,t1-t0

def build_graph(ranks):
    n_voters, n_candidates = ranks.shape
    edge_weights = np.zeros((n_candidates, n_candidates))
    for i, j in combinations(range(n_candidates), 2):
        preference = ranks[:, i] - ranks[:, j]
        h_ij = np.sum(preference < 0)  # prefers i to j
        h_ji = np.sum(preference > 0)  # prefers j to i
        if h_ij > h_ji:
            edge_weights[i, j] = h_ij - h_ji
        elif h_ij < h_ji:
            edge_weights[j, i] = h_ji - h_ij
    return edge_weights

