# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:42:32 2023

@author: 91721
"""
import pandas as pd
import matplotlib.pyplot as plt
#Now import file from data set and create a dataFrame
Univ1 = pd.read_excel("C:/Data_Set/University_Clustering.xlsx")
a = Univ1.describe()
#We have one column "State" which really not useful we will drop it
Univ = Univ1.drop(["State"] , axis=1)
#We know that there is scale difference among the columns,
#which we have to remove
#either by using normalization or standardization
#Whenever there is mixed data apply normalization

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

#Now apply this normalization fucntion to Univ dataFrame for all the rows
#Since 0th column has University name hence skipped
df_norm = norm_func(Univ.iloc[:,1:])
#you can ckeck the df_norm dataFrame which is scaled between 
#values of 0 and 1
#you can apply describe function to new data frame 
b = df_norm.describe()
#Before you apply clustering , you need to plot dendogram first
#Now to create dendogram , we need to measure distance,
#we have to import linkage
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#linkage function gives us hierarchicla or aglomerative clustering 
#ref the help for linkage
z = linkage(df_norm , method="complete" , metric="euclidean")
plt.figure(figsize = (15,8));
plt.title("Hierarchical Clustering dendogram");
plt.xlabel("Index");
plt.ylabel("Distance")
#ref help of dendogram
#sch.dendogram(z)
sch.dendrogram(z,leaf_rotation=0, leaf_font_size =10)
plt.show()
#dendogram()
#applying agglomerative clustering choosing 3 as clusters
#from dendogram
#whatever has been displayed is dendogram is not clustering
#it is just showing numbers of possible clusters

from sklearn.cluster import AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=3 , linkage="complete",affinity="euclidean").fit(df_norm)

#apply labels to the clusters
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
#Assign this series to Univ DataFrame as columns and name the columsbn
Univ['clust'] = cluster_labels
#we want to relocate the columns 7 to 0 th position
Univ1.shape
Univ1 = Univ.iloc[: , [7,1,2,3,4,5,6]]
#now check the Univ1 dataframe
Univ = Univ.iloc[:,2:].groupby(Univ1.clust).mean()
#from the output clusters 2 has got higest Top10
#lowest accept ratio, best faculty ration and higest expenses
#higest graduates ratio
Univ1.to_csv("C:/4-DataPreprosesing/University.csv" , encoding="utf8")
import os
os.getcwd()


#---------------------------------------------------------------------

