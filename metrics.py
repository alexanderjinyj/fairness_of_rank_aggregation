import numpy as np
import pandas as pd


def pairwise_statistical_parity(ranking_1, ranking_2):
    """
    compute the pairwise_statistical_parity for ranking.
    compute the pairwise_statistical_parity for ranking.
    :parameter ranking_1 is A numpy 2d_array of ranking over the value1 first column is the id of the candidates 2nd is the
    ranking value of the candidates.
    :parameter ranking_2 is A numpy 2d_array of ranking over the value1 first column is the id of
    the candidates 2nd is the ranking value of the candidates.
    
    """
    # return the parity value
    parity = 0

    for x in ranking_1:
        for y in ranking_2:
            # if in the ranking candidate x in ranking_1 is smaller than y in ranking_2 parity plus 1
            if x[1] < y[1]:
                parity += 1
            # if in the ranking candidate x in ranking_1 is greater than y in ranking_2 parity minus 1
            elif x[1] > y[1]:
                parity += -1
    parity = parity / (len(ranking_1) * len(ranking_2))
    return parity


def top_k_parity(groups, k):
    """
        computer whether the ranking with mutually exclusive groups satisfies top-k parity.
        groups are the array of  group,
        the group in the groups is a 2d array first column is the id of the candidates and
        2nd is the ranking value of the candidates.
        k is the top k.
        calculate the number of candidate(rank smaller than k) in each group then
        counter the Proportion: number of candidate(rank smaller than k)/number of group
        then calculate whether the proportion on each group is equaled if True satisfied top-k parity,
        if False not satisfied.
    """
    # result value of satisfied top-k parity
    satisfied = True
    # size of the groups
    # counter the ith group in the iteration
    # group_counter=0
    # a array, element is number of candidate(rank smaller than k) in one group
    cand_counters = []
    # counter the sum of candidate
    sum_candidates = 0
    # the counter of number of candidate in one group
    num_candidates = []
    for group in groups:
        # count the number of candidate(rank smaller than k) in this group
        cand_counter = 0
        # get the number of candidate in the group
        num_candidate = group.shape[0]
        # add the number of candidate in the group into counter
        num_candidates.append(num_candidate)
        # count the sum of candidate in the group
        sum_candidates = sum_candidates + num_candidate

        for candidate in group:
            # if the candidate´s rank value smaller than k  then cand_counter +1
            if candidate[1] <= k:
                cand_counter += 1
        # add it to counters array
        cand_counters.append(cand_counter)
    # check whether the candidate(whose rank is smaller than k) in each group is proportional to number of candidate
    # in each group
    for i in range(0, groups.shape[0]):
        # compute the ratio of the number of candidate in this group and the sum of the candidate
        proportion = num_candidates[i] / sum_candidates
        # if number of candidate is not equal to the round of the (proportion* k),then its not satisfied
        if cand_counters[i] != round(proportion * k):
            satisfied = False
            break
    return satisfied


def rank_equality_error(rank_1, rank_2, group_1, group_2):
    """
    computer the Rank Equality Error ratio the pair which one element for group_1 and another form group_2 has different
    favoring in 2 ranks. This means one has lower rank value than other in one rank but inverted in another rank
    param rank_1 is a 2d array with candidate and rank
    param rank_2 is a 2d array with candidate and rank
    param group_1 is a array with candidates
    param group_2 is a array with candidates

    """
    # count_pairs is-the number of pairs has different favoring
    count_pairs = 0
    # calculate the count_pairs
    for candidate_1 in group_1:
        for candidate_2 in group_2:
            # get the rank of each candidate in each ranking
            rank_1_of_candidate_1 = rank_1[np.argwhere(rank_1[:, 0] == candidate_1), 1]
            rank_1_of_candidate_2 = rank_1[np.argwhere(rank_1[:, 0] == candidate_2), 1]
            rank_2_of_candidate_1 = rank_2[np.argwhere(rank_2[:, 0] == candidate_1), 1]
            rank_2_of_candidate_2 = rank_2[np.argwhere(rank_2[:, 0] == candidate_2), 1]
            # if candidate1 and candidate2 has different favoring in 2 rankings then count_pairs +1
            if np.sign(rank_1_of_candidate_1 - rank_1_of_candidate_2) == np.sign(
                    rank_2_of_candidate_2 - rank_2_of_candidate_1):
                count_pairs += 1
    # the number of mix pairs
    number_pairs = group_1.shape[0] * group_2.shape[0]
    # the ratio of Rank Equality Error
    rqe = count_pairs / number_pairs
    return rqe


def attribute_rank_parity(rank, attributes, k_attribute):
    """
    calculate arp value for the protected attribute p in rank 
    param rank is a 2d array with candidate and attribute value and rank value
    param attributes is array with candidate and protected attributes
    param k_attribute is the kth attribute k
    """
    # get the all kth attributes value  +1 because attributes[0] is candidates
    group_of_values = np.unique(attributes[:, k_attribute + 1])
    # get the number of attributes in attributes prepare the parament attributes_values for function
    # favored_pair_representation
    number_of_attributes = attributes.shape[1] - 1
    # array of fpr score of all value of kth attribute
    fprs = []
    # compute teh fpr of every value of kth attribute
    for value in group_of_values:
        # make the attributes_values, nan means not the kth attribute
        attributes_values = np.full(number_of_attributes, np.nan, dtype=object)
        attributes_values[k_attribute] = value
        # compute fpr of value
        fpr = favored_pair_representation(rank, attributes, attributes_values)
        fprs.append([value, float(fpr)])
    fprs = np.array(fprs)
    # get max fprs_scores and converse to float (in fprs its string cant applied in mix and min function)
    fprs_scores = fprs[:, 1].astype(np.float64)
    # get max fprs
    max_fpr = np.max(fprs_scores)
    # get min fpr
    min_fpr = np.min(fprs_scores)
    # get the value which has greatest fpr
    max_value = fprs[np.argwhere(fprs[:, 1].astype(np.float64) == max_fpr), 0]
    # get the value which has smaller est fpr
    min_value = fprs[np.argwhere(fprs[:, 1].astype(np.float64) == min_fpr), 0]
    # compute arp
    arp = max_fpr - min_fpr
    # return a tuple of max_value ,min_value, arp
    return max_value, min_value, arp


def favored_pair_representation(rank, attributes, attributes_values):
    """
    calculate fps score param rank is a 2d array with candidate and attribute value and rank value param attributes
    is array with candidate and protected attributes param attributes_values is the array with values of attributes
    for interested attribute and the uninterested attribute's value is nan
    """

    # group of candidates whose attributes values are  equal to attributes_values
    protected_group = []
    # group of candidates whose attributes values are not equal to attributes_values
    other_group = []
    # divide candidates into 2 groups
    for attribute in attributes:
        # qualified_candidate check whether the candidates' attributes values are  equal to attributes_values
        qualified_candidate = True
        # check whether the candidates' attributes values are  equal to attributes_values
        for value in range(len(attributes_values)):
            # if the value of one attribute in attributes_values is nan dont check it
            # if its not nan check the value
            if ~pd.isnull(attributes_values[value]):
                # check value if is not equal qualified_candidate become False and end loop.because attribute is
                # tuple of (candidates and attributes value) and attributes_values is the tuple of attributes values
                # of index of  attribute should +1
                if attribute[value + 1] != attributes_values[value]:
                    qualified_candidate = False
                    break
                    # if candidate is qualified add it to qualified_candidates,if not add it to other_group
        if qualified_candidate:
            protected_group.append(attribute[0])
        else:
            other_group.append(attribute[0])

    # number of pair in mixed pairs equals to (number of candidate in protected_group*number of candidate in
    # other_group)
    mixed_pairs = len(protected_group) * len(other_group)
    # number of count pairs
    count_pairs = 0
    # compute the number of count pair
    for protected_candidate in protected_group:
        for other_candidate in other_group:
            rank_of_protected = rank[np.argwhere(rank[:, 0] == protected_candidate), 1]
            rank_of_other = rank[np.argwhere(rank[:, 0] == other_candidate), 1]
            if rank_of_protected < rank_of_other:
                count_pairs += 1
    # fpr score is the result
    fpr = count_pairs / mixed_pairs
    return fpr
