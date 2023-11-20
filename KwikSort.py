from itertools import combinations

import numpy as np

from random import randrange


def KwikSort(Vertices,rankings):
    Fas_Tournament=build_graph(rankings)
    unweighted_majority_Tournament=generate_unweighted_majority_Tournament(Vertices,Fas_Tournament)
    ranking_opt=Kwik_Sort_recur(unweighted_majority_Tournament,Fas_Tournament)

    return ranking_opt

def Kwik_Sort_recur(unweighted_majority_Tournament,Fas_Tournament):
    Vertices=unweighted_majority_Tournament[0]
    Arc=unweighted_majority_Tournament[1]
    if len(Vertices) ==0:
        return np.zeros(0)
    size_candidates=len(Vertices)
    pivot_candidate=Vertices[randrange(size_candidates)]
    Vertices_L=[]
    Vertices_R=[]
    for j in range(size_candidates):
        if Arc[Vertices[j],pivot_candidate]==1:
            Vertices_L.append(Vertices[j])
        if Arc[pivot_candidate,Vertices[j]]==1:
            Vertices_R.append(Vertices[j])
    Vertices_L=np.array(Vertices_L)
    Vertices_R=np.array(Vertices_R)
    unweighted_majority_Tournament_L=generate_unweighted_majority_Tournament(Vertices_L,Fas_Tournament)
    unweighted_majority_Tournament_R=generate_unweighted_majority_Tournament(Vertices_R,Fas_Tournament)
    permutation_L=Kwik_Sort_recur(unweighted_majority_Tournament_L,Fas_Tournament).astype(int)
    permutation_R=Kwik_Sort_recur(unweighted_majority_Tournament_R,Fas_Tournament).astype(int)
    return np.concatenate((permutation_L,np.array([pivot_candidate]).astype(int),permutation_R))


def generate_unweighted_majority_Tournament(Verticles,Fas_Tournament):
    size_candidates=Fas_Tournament.shape[0]
    size_Vertices=len(Verticles)
    Arc=np.zeros((size_candidates,size_candidates))
    for i,j in combinations(range(size_Vertices), 2):
        if Fas_Tournament[Verticles[i],Verticles[j]]>Fas_Tournament[Verticles[j],Verticles[i]]:
            Arc[Verticles[i],Verticles[j]]=1
        else:
            Arc[Verticles[j],Verticles[i]]=1
    return Verticles,Arc



def fair_Kwik_Sort(rankings,attribute,groups,threshold):
    Vertices=np.arange(rankings.shape[1])
    Fas_Tournament=build_graph(rankings)
    unweighted_majority_Tournament=generate_unweighted_majority_Tournament(Vertices,Fas_Tournament)
    ranking_opt=fair_Kwik_Sort_recur(rankings,unweighted_majority_Tournament,Fas_Tournament,attribute,groups,threshold)

    return ranking_opt

def fair_Kwik_Sort_recur(rankings,unweighted_majority_Tournament,Fas_Tournament,attribute,groups, threshold):
    Vertices=unweighted_majority_Tournament[0]
    Arc=unweighted_majority_Tournament[1]
    size_candidates=len(Vertices)
    if size_candidates ==0:
        return np.zeros(0)

    pivot_candidate=Vertices[randrange(size_candidates)]
    pivot_candidate=[pivot_candidate]
    if size_candidates==1:
        return np.array([pivot_candidate[0]])
    Vertices_L=[]
    Vertices_R=[]
    for j in range(size_candidates):
        if Arc[Vertices[j],pivot_candidate[0]]==1:
            Vertices_L.append(Vertices[j])
        if Arc[pivot_candidate[0],Vertices[j]]==1:
            Vertices_R.append(Vertices[j])

    if size_candidates==2:
        result = np.concatenate((Vertices_L, np.array([pivot_candidate[0]]).astype(int), Vertices_R))
        return result

    Vertices_L, pivot_candidate, Vertices_R=adjust_unfairness(rankings,Vertices_L,pivot_candidate,Vertices_R,groups,attribute,threshold)
    unweighted_majority_Tournament_L=generate_unweighted_majority_Tournament(Vertices_L,Fas_Tournament)
    unweighted_majority_Tournament_R=generate_unweighted_majority_Tournament(Vertices_R,Fas_Tournament)

    permutation_L=fair_Kwik_Sort_recur(rankings,unweighted_majority_Tournament_L,Fas_Tournament,attribute,groups,threshold)
    permutation_R=fair_Kwik_Sort_recur(rankings,unweighted_majority_Tournament_R,Fas_Tournament,attribute,groups,threshold)
    if len(permutation_L)>0:
        permutation_L=permutation_L.astype(int)
    if len(permutation_R)>0:
        permutation_R=permutation_R.astype(int)
    if len(pivot_candidate)>0:
        return np.concatenate((permutation_L,np.array([pivot_candidate[0]]).astype(int),permutation_R))
    else:
        return np.concatenate((permutation_L,permutation_R))

def unfairness(Vertices,boundary_0,boundary_1,groups,attribute):
    groups_of_Vertices=group_by(Vertices,groups,attribute)
    size_of_group_0=len(groups_of_Vertices[0])
    size_of_group_1=len(groups_of_Vertices[1])
    unfairness_of_group_0=(max(0,boundary_0[0]-size_of_group_0),max(0,size_of_group_0-boundary_0[1]))
    unfairness_of_group_1=(max(0,boundary_1[0]-size_of_group_1),max(0,size_of_group_1-boundary_1[1]))
    return  unfairness_of_group_0, unfairness_of_group_1


def boundary(proportion_group,length_Vertices,threshold):
    return(round((proportion_group-threshold)*(length_Vertices)),round((proportion_group+threshold)*(length_Vertices)))

def adjust_unfairness(rankings,Vertices_L,pivot_candidate,Vertices_R,groups,attribute,threshold):
    length_Vertices_L=len(Vertices_L)
    length_Vertices_R=len(Vertices_R)
    #print(length_Vertices_L,length_Vertices_R,pivot_candidate)
    proportions_groups=proportions(groups)
    boundary_L=[]
    boundary_L.append(boundary(proportions_groups[0],length_Vertices_L,threshold))
    boundary_L.append(boundary(proportions_groups[1],length_Vertices_L,threshold))
    boundary_R=[]
    boundary_R.append(boundary(proportions_groups[0],length_Vertices_R,threshold))
    boundary_R.append(boundary(proportions_groups[1],length_Vertices_R,threshold))
    #print(boundary_L,boundary_R,'boundary')
    groups_of_Vertices_L=group_by(Vertices_L,groups,attribute)
    groups_of_Vertices_R=group_by(Vertices_R,groups,attribute)
    size_of_group_L=(len(groups_of_Vertices_L[0]),len(groups_of_Vertices_L[1]))
    size_of_group_R=(len(groups_of_Vertices_R[0]),len(groups_of_Vertices_R[1]))
    #print(size_of_group_L,size_of_group_R,'size')
    unfairness_of_group_L=[]
    unfairness_of_group_L.append((max(0,boundary_L[0][0]-size_of_group_L[0]),max(0,size_of_group_L[0]-boundary_L[0][1])))
    unfairness_of_group_L.append((max(0,boundary_L[1][0]-size_of_group_L[1]),max(0,size_of_group_L[1]-boundary_L[1][1])))
    unfairness_of_group_R=[]
    unfairness_of_group_R.append((max(0,boundary_R[0][0]-size_of_group_R[0]),max(0,size_of_group_R[0]-boundary_R[0][1])))
    unfairness_of_group_R.append((max(0,boundary_R[1][0]-size_of_group_R[1]),max(0,size_of_group_R[1]-boundary_R[1][1])))
    #print(unfairness_of_group_L,unfairness_of_group_R,'unfair')
    group_pivot=attribute[pivot_candidate[0]][1]
    for group in groups:
        # print(group,'p')
        if group_pivot==group[0]:
            #print(group,group_pivot,'0p')
            gap_fairness_pivot=np.zeros(4)
            gap_fairness_pivot[0]=unfairness_of_group_L[group[0]][0]
            gap_fairness_pivot[1]=unfairness_of_group_R[group[0]][0]
            gap_fairness_pivot[2]=unfairness_of_group_L[group[0]][1]
            gap_fairness_pivot[3]=unfairness_of_group_R[group[0]][1]
            max_unfairness_pivot=np.argmax(gap_fairness_pivot)
            if gap_fairness_pivot[max_unfairness_pivot] >0:
                if max_unfairness_pivot==0:
                    kemeny_ranking_R=KwikSort(groups_of_Vertices_R[group[0]],rankings)
                    Vertices_L.append(pivot_candidate[0])
                    size_of_group_L_pivot=size_of_group_L[group[0]]+1
                    new_boundary_L_pivot=boundary(proportions_groups[group[0]],len(Vertices_L),threshold)[0]
                    for i in range(int(gap_fairness_pivot[max_unfairness_pivot] )-1):
                        #print(i,size_of_group_L_pivot,new_boundary_L_pivot,'a')
                        if size_of_group_L_pivot>=new_boundary_L_pivot:
                            break
                        Vertices_L.append(kemeny_ranking_R[i])
                        Vertices_R.remove(kemeny_ranking_R[i])
                        size_of_group_L_pivot=size_of_group_L_pivot+1
                        new_boundary_L_pivot=boundary(proportions_groups[group[0]],len(Vertices_L),threshold)[0]
                    pivot_candidate=[]

                if max_unfairness_pivot==1:
                    kemeny_ranking_L=KwikSort(groups_of_Vertices_L[group[0]],rankings)
                    Vertices_R.append(pivot_candidate[0])
                    size_of_group_R_pivot=size_of_group_L[group[0]]
                    new_boundary_R_pivot=boundary(proportions_groups[group[0]],len(Vertices_R),threshold)[0]
                    size_of_kemeny_ranking_L = len(kemeny_ranking_L)
                    for i in range(int(gap_fairness_pivot[max_unfairness_pivot] )):
                        #print(i,size_of_group_R_pivot,new_boundary_R_pivot,size_of_kemeny_ranking_L,'b')
                        if size_of_group_R_pivot>=new_boundary_R_pivot:
                            break
                        if(size_of_kemeny_ranking_L==0):
                            break
                        Vertices_R.append(kemeny_ranking_L[size_of_kemeny_ranking_L-i-1])
                        Vertices_L.remove(kemeny_ranking_L[size_of_kemeny_ranking_L-i-1])
                        size_of_group_R_pivot=size_of_group_R_pivot+1
                        new_boundary_R_pivot=boundary(proportions_groups[group[0]],len(Vertices_R),threshold)[0]
                    pivot_candidate=[]

                if max_unfairness_pivot==2:
                    kemeny_ranking_L=KwikSort(groups_of_Vertices_L[group[0]],rankings)
                    Vertices_R.append(pivot_candidate[0])
                    size_of_group_L_pivot=size_of_group_L[group[0]]
                    new_boundary_L_pivot=boundary(proportions_groups[group[0]],len(Vertices_L),threshold)[1]
                    size_of_kemeny_ranking_L = len(kemeny_ranking_L)
                    for i in range(int(gap_fairness_pivot[max_unfairness_pivot] )):
                        #print(i,size_of_group_L_pivot,new_boundary_L_pivot,size_of_kemeny_ranking_L,'c')
                        if size_of_group_L_pivot<=new_boundary_L_pivot:
                            break
                        if(size_of_kemeny_ranking_L==0):
                            break
                        Vertices_R.append(kemeny_ranking_L[size_of_kemeny_ranking_L-i-1])
                        Vertices_L.remove(kemeny_ranking_L[size_of_kemeny_ranking_L-i-1])
                        size_of_group_L_pivot=size_of_group_L_pivot-1
                        new_boundary_L_pivot=boundary(proportions_groups[group[0]],len(Vertices_L),threshold)[1]
                    pivot_candidate=[]

                if max_unfairness_pivot==3:
                    kemeny_ranking_R=KwikSort(groups_of_Vertices_R[group[0]],rankings)
                    Vertices_L.append(pivot_candidate[0])
                    size_of_group_R_pivot=size_of_group_R[group[0]]
                    new_boundary_R_pivot=boundary(proportions_groups[group[0]],len(Vertices_R),threshold)[1]
                    for i in range(int(gap_fairness_pivot[max_unfairness_pivot] )):
                        #print(i,size_of_group_R_pivot,new_boundary_R_pivot,'d')
                        if size_of_group_R_pivot<=new_boundary_R_pivot:
                            break
                        Vertices_L.append(kemeny_ranking_R[i])
                        Vertices_R.remove(kemeny_ranking_R[i])
                        size_of_group_R_pivot-=1
                        new_boundary_R_pivot=boundary(proportions_groups[group[0]],len(Vertices_R),threshold)[1]
                    pivot_candidate=[]

        else:
            gap_fairness_non_pivot=np.zeros(4)
            gap_fairness_non_pivot[0]=unfairness_of_group_L[group[0]][0]
            gap_fairness_non_pivot[1]=unfairness_of_group_R[group[0]][0]
            gap_fairness_non_pivot[2]=unfairness_of_group_L[group[0]][1]
            gap_fairness_non_pivot[3]=unfairness_of_group_R[group[0]][1]
            max_unfairness_non_pivot=np.argmax(gap_fairness_non_pivot)

            if max_unfairness_non_pivot>=0:

                if gap_fairness_non_pivot[max_unfairness_non_pivot]==0:
                    kemeny_ranking_R=KwikSort(groups_of_Vertices_R[group[0]],rankings)
                    size_of_group_L_non_pivot=size_of_group_L[group[0]]
                    for i in range(int(gap_fairness_non_pivot[max_unfairness_non_pivot])):
                        #print(i,size_of_group_L_non_pivot,kemeny_ranking_R,'e')
                        Vertices_L.append(kemeny_ranking_R[i])
                        Vertices_R.remove(kemeny_ranking_R[i])
                        size_of_group_L_non_pivot+=1
                        new_boundary_L_non_pivot=boundary(proportions_groups[group[0]],len(Vertices_L),threshold)[0]
                        if size_of_group_L_non_pivot >= new_boundary_L_non_pivot:
                            break


                if max_unfairness_non_pivot==1:
                    kemeny_ranking_L=KwikSort(groups_of_Vertices_L[group[0]],rankings)
                    size_of_group_R_non_pivot=size_of_group_L[group[0]]
                    size_of_kemeny_ranking_L = len(kemeny_ranking_L)
                    for i in range(int(gap_fairness_non_pivot[max_unfairness_non_pivot])):
                        # print(i,size_of_group_R_non_pivot,size_of_kemeny_ranking_L,'f')
                        if(size_of_kemeny_ranking_L==0):
                            break
                        Vertices_R.append(kemeny_ranking_L[size_of_kemeny_ranking_L-i-1])
                        Vertices_L.remove(kemeny_ranking_L[size_of_kemeny_ranking_L-i-1])
                        size_of_group_R_non_pivot=size_of_group_R_non_pivot+1
                        new_boundary_R_non_pivot=boundary(proportions_groups[group[0]],len(Vertices_R),threshold)[0]
                        # print(i,size_of_group_R_non_pivot,size_of_kemeny_ranking_L,new_boundary_R_non_pivot,'f')
                        if size_of_group_R_non_pivot>=new_boundary_R_non_pivot:
                            break




                if max_unfairness_non_pivot==2:
                    kemeny_ranking_L=KwikSort(groups_of_Vertices_L[group[0]],rankings)
                    size_of_group_L_non_pivot=size_of_group_L[group[0]]
                    size_of_kemeny_ranking_L = len(kemeny_ranking_L)
                    for i in range(int(gap_fairness_non_pivot[max_unfairness_non_pivot])):
                        size_of_kemeny_ranking_L = len(kemeny_ranking_L)
                        #print(i,size_of_group_L_non_pivot,size_of_kemeny_ranking_L,'g')
                        if(size_of_kemeny_ranking_L==0):#only 1 element situation
                            break
                        Vertices_R.append(kemeny_ranking_L[size_of_kemeny_ranking_L-i-1])
                        Vertices_L.remove(kemeny_ranking_L[size_of_kemeny_ranking_L-i-1])
                        new_boundary_L_non_pivot=boundary(proportions_groups[group[0]],len(Vertices_L),threshold)[1]
                        #print(i,size_of_group_L_non_pivot,new_boundary_L_non_pivot,size_of_kemeny_ranking_L,'g')
                        if size_of_group_L_non_pivot <= new_boundary_L_non_pivot:
                            break



                if max_unfairness_non_pivot==3:
                    kemeny_ranking_R=KwikSort(groups_of_Vertices_R[group[0]],rankings)
                    size_of_group_R_non_pivot=size_of_group_R[group[0]]
                    for i in range(int(gap_fairness_non_pivot[max_unfairness_non_pivot])):
                        #print(i,size_of_group_R_non_pivot,'h')
                        Vertices_L.append(kemeny_ranking_R[i])
                        Vertices_R.remove(kemeny_ranking_R[i])
                        size_of_group_R_non_pivot-=1
                        new_boundary_R_non_pivot=boundary(proportions_groups[group[0]],len(Vertices_R),threshold)[1]
                        if size_of_group_R_non_pivot<=new_boundary_R_non_pivot:
                            break



    return Vertices_L, pivot_candidate, Vertices_R





def group_by(Vertices,groups,attribute):
    groups_by=[]
    for values in groups:
        group=[]
        groups_by.append(group)
    for vertex in Vertices:
        group_of_Vertice=attribute[vertex]
        groups_by[group_of_Vertice[1]].append(vertex)
    return groups_by

def proportions(groups):
    proportions = []
    total_number = np.sum(groups[:, 1])
    for group in groups:
        proportions.append(group[1] / total_number)
    proportions = np.array(proportions)
    return proportions



def build_graph(rankings):
    n_voters, size_candidate = rankings.shape
    edge_weights = np.zeros((size_candidate, size_candidate))
    for i, j in combinations(range(size_candidate), 2):
        preference = rankings[:, i] - rankings[:, j]
        preference_i_j = np.sum(preference < 0)  # prefers i to j
        preference_j_i = np.sum(preference > 0)  # prefers j to i
        if preference_i_j > preference_j_i:
            edge_weights[i, j] = preference_i_j 
        elif preference_i_j < preference_j_i:
            edge_weights[j, i] = preference_j_i
    return edge_weights