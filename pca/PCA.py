from PIL import Image

import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#partie 1
img = Image.open(r'D:\TEK UP 2eme\trait analytique avec python\yalefaces\subject01.glasses')

import matplotlib.pyplot as plt
plt.imshow(img,'gray')



img1=np.array(img)
img2=img1.reshape(img1.shape[0]*img1.shape[1])



import glob
path=glob.glob(r'D:\TEK UP 2eme\trait analytique avec python\yalefaces\subject*')

l=[]
for p in path:
    img=Image.open(p)
    img1=np.array(img)
    img2=img1.reshape(img1.shape[0]*img1.shape[1])
    l.append(img2)
    
data= np.array(l)    



y=[]
y=np.repeat(range(1,16),11)




from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

df=pd.concat([pd.DataFrame(data),pd.DataFrame(y)],axis=1)
# suffle data
df.sample(frac=1) 




def calcul(n):
    pca = PCA(n_components=n) 
    x = pca.fit_transform(df.iloc[:,:-1])
    x_train, x_test, y_train, y_test = train_test_split(x, df.iloc[:,-1], test_size=0.33, random_state=42)
    clf = svm.SVC(kernel='linear')
    clf.fit(x_train, y_train)
    pred =clf.predict(x_test)
    acc =accuracy_score(y_test, pred)
    print("for componeents= ",n," on a accuracy = ", acc)
    
n_comp=[165,100,50,25,15]
for i in n_comp:
    calcul(i)