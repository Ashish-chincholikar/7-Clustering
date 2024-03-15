# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 08:42:27 2023

@author: Ashish Chincholikar
K-means clustering
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#let us try to understand first how k means works fro two 
#dimensional data
#for that, generate random numbers in the range 0 to 1
#and with uniform probability of 1/50

X = np.random.uniform(0,1,50)
Y = np.random.uniform(0,1,50)
#here 50 random numbers are generated ranging from 0 to 1

#create a empty dataFrame with 0 rows and 2 columns
df_xy = pd.DataFrame(columns=["X","Y"])


#assign the values of X and Y to these columns
df_xy.X = X
df_xy.Y = Y

#ploting a scatter plot so as to have a visual interpretation of the 
#data point in our dataFrame
df_xy.plot(x = "X", y = "Y" , kind = "scatter")
model1 = KMeans(n_clusters=3).fit(df_xy)

""" 
with data X and Y , apply Kmeans model,
generate scatter plot
with scale/font=10
cmap = plt.cm.coolwarm:cool color cobination
"""
#model1.labels_ --> what does this do

model1.labels_
# labels are identified so as to which the datapoints are grouped to
#labels are 0,1,2

df_xy.plot(x = "X" , y="Y" , c=model1.labels_ , kind="scatter" , s=10 , cmap=plt.cm.coolwarm)
#here the scatter plot is show which has 3 clusters

Univ1 = pd.read_excel("C:/Data_Set/University_Clustering.xlsx")
Univ1.describe()
Univ = Univ1.drop(["State"] , axis = 1)
#we know that there is scale difference among the columns , which we have 
#either by using normalization or standardization

def norm_fun(i):
    x = (i-i.min())/(i.max()-i.min())
    return x

#Now apply this normalization function to Univ dataframe for all teh 
#rows
df_norm = norm_fun(Univ.iloc[:,1:])

'''
What will be ideal cluster number, will it be 1,2,3 
iterating for all the values of k and making the elbow curve
'''
TWSS = []
k = list(range(2,8))
for i in k :
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df_norm)

    TWSS.append(kmeans.inertia_)#total within sum of square

'''
Kmeans inertia, also known as sum of squares Errors or(SSE) ,
calculates the sum of the distance of all points within a cluster
form the centroid of the point , it is the difference betweent the
observed value and the predicted value. 
'''
    
TWSS
#AS k value increases the TWSS value decreases

plt.plot(k,TWSS,'ro-');
plt.xlabel("No_of_clusters");
plt.ylabel("Total_Within_SS")

'''
How to select value of k from elbow curve
when k changes from 2 to 3 , then decreases in twss
in higher than when k changes from 3 to 4
When k values changes from 5 to 6 decrease in twss
is considerably less, hence considered k=3 
'''

model = KMeans(n_clusters=3) #decided the value of k
model.fit(df_norm) #making the dataframe to fit the model
model.labels_#assigning the labels
mb = pd.Series(model.labels_)
Univ['clust'] = mb
Univ.head()
Univ = Univ.iloc[: , [7,0,1,2,3,4,5,6]]
Univ
Univ.iloc[:,2:8].groupby(Univ.clust).mean()
Univ.to_csv("Kmeans_University.csv",encoding="utf-8")
import os
os.getcwd()

















