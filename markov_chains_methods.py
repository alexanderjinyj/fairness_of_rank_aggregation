from itertools import combinations
import queue
import numpy as np
from lief import Object
import utils as ul

"""
def aggregate_rank_first(rankings, type):
    index_candidate = 0
    size_candidate = rankings.shape[1]
    aggregate_rank_first = np.zeros(size_candidate)
    ranking_drop_i = rankings
    for i in range(size_candidate):
        index_candidate = get_first(ranking_drop_i, type)
        aggregate_rank_first[index_candidate] = i
        index_delete = np.argwhere(ranking_drop_i[0:] == index_candidate)
        ranking_drop_i = np.delete(ranking_drop_i, index_delete[0, 1], 1)
    return aggregate_rank_first


def get_first(rankings, type):
    pure_ranking = np.delete(rankings, 0, 0)
    stationary_distribution = stationary_distribute(pure_ranking, type)
    print(stationary_distribution)
    temp_ranking = np.argsort(stationary_distribution)

    first = rankings[0][temp_ranking[-1]]
    return first
"""

def aggregate_rank_mc(rankings,type):
    stationary_distribution = stationary_distribute(rankings,type)
    temp_ranking = np.argsort(stationary_distribution)
    aggregate_ranking = np.empty_like(temp_ranking)
    aggregate_ranking[temp_ranking] = np.arange(len(stationary_distribution))
    return aggregate_ranking


def stationary_distribute(rankings, type):
    transition_matrix = []
    if type == 0:
        transition_matrix = generate_transition_matrix(rankings)
    if type == 1:
        transition_matrix = generate_transition_matrix_without_self_circle(rankings)
    # transition_matrix = transition_matrix*(1-alpha)+(alpha/transition_matrix.shape[0])
    transition_matrix_trans = transition_matrix.T
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix_trans)
    close_to_1_idx = np.isclose(eigenvalues,1, rtol=0.001)
    target_eigenvector = eigenvectors[:,close_to_1_idx]
    target_eigenvector = target_eigenvector[:,0]
# Turn the eigenvector elements into probabilities
    stationary_distribute = target_eigenvector / sum(target_eigenvector)
    return stationary_distribute

def generate_transition_matrix_without_self_circle(rankings):
    size_candidate = rankings.shape[1]
    transition_matrix = np.zeros(shape=(size_candidate, size_candidate))
    preference=np.zeros(shape=(size_candidate, size_candidate))
    for i in range(size_candidate):
        for j in range(size_candidate):
            preference[i,j]=count_preference(rankings,i,j)
    for i in range(size_candidate):
        preference_i=(np.sum(preference[i,:]))
        if preference_i >0:
            for j in range(size_candidate):
                transition_matrix[i,j]=preference[i,j]/preference_i
        else:
            transition_matrix[i,:]=0
            transition_matrix[i,i]=1
    return  transition_matrix


def generate_transition_matrix(rankings):
    size_candidate = rankings.shape[1]
    transition_matrix = np.zeros(shape=(size_candidate, size_candidate))
    for i in range(size_candidate):
        for j in range(size_candidate):
            probability=count_probability_edge(rankings, i, j)
            transition_matrix[i,j] =probability
    for i in range(size_candidate):
        transition_matrix[i,i] = 1 - np.sum(transition_matrix[i, :])
    return transition_matrix

def count_probability_edge(rankings, i, j):
    count_preference = 0
    total_pairs = rankings.shape[0] * rankings.shape[1]
    for ranking in rankings:
        if ranking[i] < ranking[j]:
            count_preference += 1
    return count_preference / total_pairs


def count_preference(rankings,i,j):
    preference=np.sum((rankings[:,i]-rankings[:,j])<0)
    return preference

def mean_distance_transition_matrix(rankings):
    size_candidate = rankings.shape[1]
    transition_matrix = np.zeros(shape=(size_candidate, size_candidate))
    for i in range(size_candidate):
        for j in range(size_candidate):
            probability=count_probability_edge(rankings, i, j)
            mean_distance=mean_distance_i_j(rankings,i,j)
            if probability>0:
                transition_matrix[i,j] =probability/mean_distance
    for i in range(size_candidate):
        transition_matrix[i,i] = 1 - np.sum(transition_matrix[i, :])
    return transition_matrix

def mean_distance_i_j(rankings,i,j):
    mean_distance=0
    counter=0
    for ranking in rankings:
        if ranking[i] < ranking[j]:
            mean_distance+=ranking[j]-ranking[i]
            counter+=1
    if counter>0:
        mean_distance=mean_distance/counter
    else:
        mean_distance=0
    return mean_distance



def proportions(groups):
    proportions = []
    total_number = np.sum(groups[:, 1])
    for group in groups:
        proportions.append(group[1] / total_number)
    proportions = np.array(proportions)
    return proportions

def get_initial(transition_matrix):
    initial=np.inf
    max=0
    for i in range(transition_matrix.shape[0]):
        sum_preference= np.sum(transition_matrix[i,:])-transition_matrix[i,i]
        if sum_preference>max:
            max=sum_preference
            initial=i
    return initial



def fair_kemeny_mc_rankings_greedy(rankings, attribute, groups, threshold,type):
    group_proportions = proportions(groups)
    size_candidate = rankings.shape[1]
    transition_matrix=np.zeros(size_candidate)
    if type==0:
        transition_matrix = generate_transition_matrix(rankings)
    if type==1:
        transition_matrix= mean_distance_transition_matrix(rankings)
    fair_kemeny_rankings = []
    initial =get_initial(transition_matrix)
    print('initial: '+ str(initial))
    fair_kemeny_ranking = generate_fair_kemeny_mc_ranking_greedy(
            rankings, attribute, groups, threshold, group_proportions, transition_matrix, initial)
    fair_kemeny_ranking_distance=ul.kemeny_dist(rankings,fair_kemeny_ranking)
    fair_kemeny_rankings.append((fair_kemeny_ranking,fair_kemeny_ranking_distance))
    return fair_kemeny_rankings


def generate_fair_kemeny_mc_ranking_greedy(rankings, attribute, groups, threshold, group_proportions,
                                                 transition_matrix, initial):
    size_candidate = rankings.shape[1]
    fair_kemeny_ranking = np.zeros(size_candidate)
    candidate_visited = np.zeros(size_candidate)
    candidate_visited[initial] = 1
    group_in_prefix = np.zeros(groups.shape[0])
    group_in_prefix[attribute[initial, 1]] += 1
    floor = group_proportions - threshold
    floor=np.where(floor<0,0,floor)
    ceil = group_proportions + threshold
    ceil=np.where(ceil>size_candidate,size_candidate,ceil)
    current_candidate = initial
    for steps in range(1, size_candidate):
        degree_candidate = transition_matrix[current_candidate, :]
        group_queues = generate_group_queues(degree_candidate, candidate_visited, attribute, groups)
        lower_boundary = np.floor(floor * (steps + 1))
        upper_boundary = np.ceil(ceil * (steps + 1))
        lower_tight = lower_tightness(group_in_prefix, lower_boundary)
        upper_tight = upper_tightness(group_in_prefix, upper_boundary)
        #print(list(group_queues[0].queue),list(group_queues[1].queue))
        #print(lower_tight,upper_tight,group_in_prefix,lower_boundary,upper_boundary)
        if len(lower_tight) > 0:
            lower_tight_group = lower_tight[0]
            candidate_of_lower_queue_pop = group_queues[lower_tight_group].get()
            if candidate_of_lower_queue_pop < current_candidate:
                if group_queues[lower_tight_group].qsize()>0:
                    candidate_of_lower_queue_pop = group_queues[lower_tight_group].get()
            fair_kemeny_ranking[candidate_of_lower_queue_pop] = steps
            #current_candidate = candidate_of_lower_queue_pop
            group_in_prefix[lower_tight_group] += 1
            candidate_visited[candidate_of_lower_queue_pop]=1
            group_queues[lower_tight_group].task_done()
        else:
            max_candidate_of_queues = []
            for group_index in range(groups.shape[0]):
                if len(upper_tight) > 0 and group_index == upper_tight[0]:
                    continue
                if group_queues[group_index].qsize()>0:
                    max_candidate_of_queues.append(group_queues[group_index].get())
                    group_queues[group_index].task_done()
            value_edge = degree_candidate[max_candidate_of_queues]
            max_candidate_queue_edge = np.concatenate((max_candidate_of_queues, value_edge), axis=0).reshape(2,
                                                                                                             len(max_candidate_of_queues))

            max_candidate =int (max_candidate_queue_edge[0, np.argmax(max_candidate_queue_edge[1, :])])
            current_candidate = max_candidate
            fair_kemeny_ranking[max_candidate] = steps
            group_in_prefix[attribute[max_candidate][1]] += 1
            candidate_visited[max_candidate]=1
    return fair_kemeny_ranking


def lower_tightness(group_in_prefix, lower_boundary):
    tight_groups = []
    for group_index in range(len(group_in_prefix)):
        if group_in_prefix[group_index] < lower_boundary[group_index]:
            tight_groups.append(group_index)
    return tight_groups


def upper_tightness(group_in_prefix, upper_boundary):
    tight_groups = []
    for group_index in range(len(group_in_prefix)):
        if group_index >= upper_boundary[group_index]:
            tight_groups.append(group_index)
    return tight_groups


def generate_group_queues(degree_candidate, candidate_visited, attribute, groups):
    edge_sort = np.argsort(degree_candidate)[::-1]
    group_queues = []
    for i in range(len(groups)):
        group_queue = queue.Queue()
        group_queues.append(group_queue)
    for i in edge_sort:
        if candidate_visited[i] == 0:
            group_queues[attribute[i][1]].put(i)
    return group_queues
