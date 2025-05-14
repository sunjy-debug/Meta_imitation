import metis
import networkx as nx
import pickle as pkl
import numpy as np
import scipy.io as sio
from datatools import *




with open('../Flickr/Flickr_edges.pkl','rb') as f:
    edgeList = pkl.load(f)
print (np.array(edgeList).shape)
G = nx.Graph()
G.add_edges_from(edgeList)
(cut, parts) = metis.part_graph(G, 3)
parts_save = {
        'cut':cut,
        'parts':parts
        }

trainIndex,valIndex,testIndex = dataSplit(parts_save)

data = sio.loadmat("../Flickr/Flickr0.mat")
A = data["Network"].toarray()
X = data["Attributes"]
print ("X",X.shape)

trainA = np.array([a[trainIndex] for a in A[trainIndex]])
valA = np.array([a[valIndex] for a in A[valIndex]])
testA = np.array([a[testIndex] for a in A[testIndex]])
print ("Shape of adj matrix train:{}, val:{}, test:{}".format(trainA.shape,valA.shape,testA.shape))
sumtrA = np.sum(trainA,1)
sumvlA = np.sum(valA,1)
sumttA = np.sum(testA,1)
print (np.sum(sumtrA==0)+np.sum(sumvlA==0)+np.sum(sumttA==0))
print (np.sum(sumtrA)+np.sum(sumvlA)+np.sum(sumttA))
to_remove =[]
for i in range(len(sumtrA)):
    if sumtrA[i]==0:
        to_remove.append(trainIndex[i])
for i in range(len(sumvlA)):
    if sumvlA[i]==0:
        to_remove.append(valIndex[i])
for i in range(len(sumttA)):
    if sumttA[i]==0:
        to_remove.append(testIndex[i])
print(len(to_remove))
to_remove = set(to_remove)
num,col = A.shape
print (A.shape)
newGraphid = []
for i in range((num)):
    if i not in to_remove:
        newGraphid.append(i)
print (len(newGraphid))
print(len(set(newGraphid) | set(to_remove)))
# print(trainIndex)

newX = X[newGraphid]
newA = np.array([a[newGraphid] for a in A[newGraphid]])
print (newX.shape)
print (newA.shape)
sio.savemat('../Flickr/Flickr01.mat',{'Attributes': newX,'Network':newA})

new_parts = []
for i in range(len(parts)):
    if i not in to_remove:
        new_parts.append(parts[i])
print (len(new_parts))
print (new_parts[-15:])
print (np.array(parts)[newGraphid][-15:])
parts_save = {
        'cut':cut,
        'parts':new_parts
        }
with open('../Flickr/Flickr_parts.pkl','wb') as f:
    pkl.dump(parts_save,f)

