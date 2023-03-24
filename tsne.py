import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Importing features.csv and trainLabels.csv
# load the features in completeFeaturesTrainDataset.csv
features = pd.read_csv('completeFeaturesTrainDataset.csv')

#print(features.head())
#print(features['Class'])
#print(features['ID'])

# saving classes and idx as a single dataframe (data)
data = pd.DataFrame()
data['Id'] = features['ID']
data['Class'] = features['Class']


# drop the first column (index)
features = features.drop(features.columns[0], axis=1)
# drop the first column (id)
features = features.drop(features.columns[0], axis=1)
# drop the last column (class)
features = features.drop(features.columns[-1], axis=1)

print(data.head())


#tsne = TSNE().fit_transform(features)

#plt.scatter(tsne[:,0], tsne[:,1], c=data['Class'], cmap=plt.cm.get_cmap("jet", 9))
#plt.colorbar(ticks=range(10))
#plt.show()

# as tsne hasnt shown itself to be a good method, we will try PCA

#pca = PCA(n_components=2).fit_transform(features)
#
#plt.scatter(pca[:,0], pca[:,1], c=data['Class'], cmap=plt.cm.get_cmap("jet", 9))
#plt.colorbar(ticks=range(10))
#plt.show()

# as PCA hasnt shown itself to be a good method, we will try UMAP
#import umap
#import numpy as np
#
#print("starting umap")
#
#reducer = umap.UMAP()
#embedding = reducer.fit_transform(features)
#
#print("finished umap")
#
#plt.scatter(embedding[:,0], embedding[:,1], c=data['Class'], cmap=plt.cm.get_cmap("jet", 9))
#plt.colorbar(ticks=range(10))
#plt.show()

tsne = TSNE(n_components=2, perplexity=25, random_state=42,init='pca').fit_transform(features)

plt.scatter(tsne[:,0], tsne[:,1], c=data['Class'], cmap=plt.cm.get_cmap("jet", 9))
plt.colorbar(ticks=range(10))
plt.show()




