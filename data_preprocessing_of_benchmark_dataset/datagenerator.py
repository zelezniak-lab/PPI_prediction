import os
import sys
import random
import copy

### Some code snippet taken from benchmark_eval_paper

# Function for generating random pairs from a list

def genRandomPairs(list_of_proteins,num_of_pairs, pairs_excluding_list = []):

    """
    
    """

    pairs_excluding_list = set(pairs_excluding_list)
    return_list_of_protein_pairs = set()

    while len(return_list_of_protein_pairs) < num_of_pairs:
        x =   tuple(sorted(random.sample(list_of_proteins,2)))

        if x not in pairs_excluding_list and x not in return_list_of_protein_pairs and x[0] != x[1]:
            return_list_of_protein_pairs.add(x)

    return list(return_list_of_protein_pairs)


#generate random pairs of proteins with one protein from each of two lists, skipping those in an excluded set
def genRandomPairsAB(proteinLstA,proteinLstB,numPairs,exclusionLst = []):
	exclusionSet = set(exclusionLst)
	retLst = set()
	while len(retLst) < numPairs:
		a = random.sample(proteinLstA,1)[0]
		b = random.sample(proteinLstB,1)[0]
		x = tuple(sorted([a,b]))
		if x not in exclusionSet and x not in retLst and x[0] != x[1]:
			retLst.add(x)
	return list(retLst)



#split a set of proteins into groups randomly, and return a list of the groups, and dictionary mapping proteins to group IDs
def createProteinGroups(proteinLst,numGroups):
	proteinLst = list(set(proteinLst)) #make copy before shuffling to not alter original
	random.shuffle(proteinLst)
	retLst = []
	for i in range(0,numGroups):
		start = (len(proteinLst)* i)//numGroups
		end = (len(proteinLst)* (i+1))//numGroups
		if i == numGroups-1:
			end = len(proteinLst)
		retLst.append(proteinLst[start:end])
	retDict = {}
	for i in range(0,len(retLst)):
		for item in retLst[i]:
			retDict[item] = i
	return retLst, retDict


#given a list of pairs, as well as a list and dictionary of groups, creates a dictionary of dictionary of the N groups, and mapped protein pairs to the appropriate double indexed dictionary
def assignPairsToGroups(pairLst,groupsLst,groupDict):
    
    pairGroups = []

    for i in range(0,len(groupsLst)):
        pairGroups.append([])
        for j in range(0,len(groupsLst)):
            pairGroups[-1].append(set())

    for item in pairLst:
        idx1 = groupDict.get(item[0],-1)
        idx2 = groupDict.get(item[1],-1)
        if idx1 == -1:
            print('Error, cannot find item ',item[0])
            exit(42)
        if idx2 == -1:
            print('Error, cannot find item ',item[1])
            exit(42)
        
        m = min(idx1,idx2)
        m2 = max(idx1,idx2)
        pairGroups[m][m2].add(item)

    return pairGroups


#given a list of pairs, a number of pairs to draw, and a list of pairs to exclude, draws num pairs without repeating, and returns those as a new list
#assumes numPairs < len(pairLst)
def drawPairs(pairLst,numPairs,skipLst = []):
    if len(skipLst) > 0:
        pairLst = list(set(pairLst)-set(skipLst))
    else:
        pairLst = set(pairLst)
    
    random.shuffle(pairLst)
    return pairLst[0:numPairs]


    



