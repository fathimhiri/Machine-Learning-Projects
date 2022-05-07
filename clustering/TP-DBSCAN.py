from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
x,y=make_blobs(n_samples=600,n_features =2,centers=4)
plt.scatter(x[:,0],x[:,1])


import numpy as np
from sklearn.cluster import DBSCAN

db=DBSCAN(eps=2,min_samples=5)
model=db.fit(x)
label=model.labels_
plt.scatter(x[:,0],x[:,1],c=label)


#pour verifier qu'on a pas de noise
no_clusters=len(np.unique(label))
no_noise = list(label).count(-1)

print("estimated no. of clusters: ", no_clusters)
print("estimated no. of noise points: ", no_noise )


"hierarchical clustering"
from scipy.cluster.hierarchy import dendrogram, linkage


plt.figure(figsize =(6, 6)) 
plt.title('Dendograms') 
z=linkage(x, method ='ward')
Dendrogram = dendrogram(z ) 
plt.axhline(y=70, c='r', lw=3, linestyle='dashed')




"agglomerative clutsering"
from sklearn.cluster import AgglomerativeClustering 
cl = AgglomerativeClustering (n_clusters = 4,affinity = "euclidean",linkage="ward")
YPred = cl.fit_predict(x)


plt.figure(figsize=(10,7))
plt.scatter(x[:,0],x[:,1],c=YPred,cmap='plasma')
























