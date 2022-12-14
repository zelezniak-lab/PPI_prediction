import os
import sys
import random
import copy
import PPIPUtils

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

def writePosNegData(fname,pos,neg,randomOrder=True):
	lst = []
	for item in pos:
		lst.append((item[0],item[1],1))
	for item in neg:
		lst.append((item[0],item[1],0))
	if randomOrder:
		random.shuffle(lst)
	else:
		lst.sort()
	PPIPUtils.writeTSV2DLst(fname,lst)
    

#generate numPos and numNeg positive and negative pairs randomly, create k folds of data, (optionally) create a folder=foldername, and save the folds to files in the folder
#folderName should end in '/' or '\\'
#intLst and protLst are known interactions (tuple pairs, where for each pair (X,Y), X < Y), and protLst is the list of all proteins
def createRandomKFoldData(intLst, protLst,numPos,numNeg,k,folderName):
	PPIPUtils.makeDir(folderName)
	pos = drawPairs(intLst,numPos)
	neg = genRandomPairs(protLst,numNeg,intLst)
	trainSets, testSets = PPIPUtils.createKFolds(pos,neg,k,seed=None)
	for i in range(0,k):
		PPIPUtils.writeTSV2DLst(folderName+'Train_'+str(i)+'.tsv',trainSets[i])
		PPIPUtils.writeTSV2DLst(folderName+'Test_'+str(i)+'.tsv',testSets[i])

#Primary function for creating non-heldout train and test data
#generats random splits of non-overlapping train and test data, and writes the groups to a file in the given folder
#function generates 1 folder, X train set, and X*N test sets where n is the length of the 2nd argument of tuple pairs
#intLst and protLst are known interactions (tuple pairs, where for each pair (X,Y), X < Y), and protLst is the list of all proteins
#ratioslst a tuple contiaing the following information in the format (A,B,F),[(C1,D1,F1)...]):
#lst tuple arg1, tuple (A,B,F)
#A -- number of positive pairs in training data, integer
#B -- number of negative pairs in training data, integer
#F -- file prefix for training data
#lst tuple arg2, lst of tuples [(C1,D1,E1),(C2,D2,E2),(C3,D3,E3). . .]
#C1,2,3... number of positive pairs in test data set (1,2,3...)
#D1,2,3... number of negative pairs in test data set (1,2,3...)
#E1,2,3... file prefix for test data set (1,2,3...)
#numSets is the number of times to iterate the ratiosLst (generating numSets train, and numSets *len(arg2) test sets)
#folderName is the name of the folder to save the data to


def createRandomData(intLst, protLst, ratiosLst,numSets,folderName):
	PPIPUtils.makeDir(folderName)
	for k in range(0,numSets):
		#generate train data
		trainPosPairs = ratiosLst[0][0]
		trainNegPairs = ratiosLst[0][1]
		trainFNamePrefix = ratiosLst[0][2]
		trainPos = drawPairs(intLst,trainPosPairs)
		trainNeg = genRandomPairs(protLst,trainNegPairs,intLst)
		writePosNegData(folderName+trainFNamePrefix+str(k)+'.tsv',trainPos, trainNeg)
		
		#generate the test data:
		for item in ratiosLst[1]:
			testPosPairs = item[0]
			testNegPairs = item[1]
			testFNamePrefix = item[2]
			testPos = drawPairs(intLst,testPosPairs,trainPos)
			testNeg = genRandomPairs(protLst,testNegPairs,intLst+trainNeg)
			writePosNegData(folderName+testFNamePrefix+str(k)+'.tsv',testPos, testNeg)
			



#Primary function for spliting proteins in groups
#splits proteins into numGroups groups, and creates a grid of which interactions fall into which groups
#continues randomly generating groups until the minimum size requirements are met
#minSizeSmall is the minimum size for group pairs where i=j, such as Group (0,0)
#minSizeLarge is the minimum size for group pairs where i!=j, such as Group (0,1)
#if you set the minimum sizes too large for the given number of groups, this functon will fail and return None, None, None
#since this will generate numGroups small groups, and (numGroups*numGroups-numGroups)/2 large groups,
#we recommend not setting minsizeSmall >= len(intLst)/((numGroups*numGroups-numGroups)/2+numGroups/2)/2 * .97, and not setting minSizeLarge more than twice that
#For example, given 120,000 interactions, and 6 groups, points will be split into 15 large groups and 6 small/half-size groups
#Given 120,000 interactions, and that 120,000/18/2*.97  = 3233.  Keeping the numbers below 3233 (small) and 6466 (large) is recommended, but lower numbers could be needed to converge

#returns list of list of proteins per group, dictionary mapping proteins to groups, and list of list of group pairs with sets of interactions



def createGroupData(intLst,protLst,numGroups,minSizeSmall,minSizeLarge,maxAttempts=1000):
	groups = []
	groupDict = {}
	groupInts = []
	for attempt in range(0,maxAttempts):
		#create groups, returning list of groups, and dictionary mapping proteins to groups
		groups, groupDict = createProteinGroups(protLst,numGroups)
		smallest1 = minSizeSmall
		smallest2 = minSizeLarge
		#split all interaction pairs into group pairs, returning list of list of groups
		groupInts = assignPairsToGroups(intLst,groups,groupDict)
		for i in range(0,len(groupInts)):
			for j in range(i,len(groupInts[i])):
				if i == j:
					smallest1 = min(smallest1,len(groupInts[i][j]))
				else:
					smallest2 = min(smallest2,len(groupInts[i][j]))	
		#ensure smallest groups meet size requirements
		if smallest1 >= minSizeSmall and smallest2 >=minSizeLarge:
			return groups, groupDict, groupInts
	else:
		return None, None, None