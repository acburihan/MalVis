import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import umap


features = pd.read_csv('completeFeaturesTrainDataset.csv')
data = pd.DataFrame()
data['Id'] = features['ID']
data['Class'] = features['Class']

# drop the first column (index)
features = features.drop(features.columns[0], axis=1)
# drop the first column (id)
features = features.drop(features.columns[0], axis=1)
# drop the last column (class)
features = features.drop(features.columns[-1], axis=1)


classes = ['Ramnit', 'Lollipop', 'Kelihos_ver3', 'Vundo', 'Simda', 'Tracur', 'Kelihos_ver1', 'Obfuscator.ACY', 'Gatak']
plots, axes = plt.subplots(1, 3, figsize=(15, 4))

# Calculating PCA
pca = PCA(n_components=2).fit_transform(features)
ax1 = axes[0]
ax1.scatter(pca[:,0], pca[:,1], c=data['Class'], cmap=plt.cm.get_cmap("jet", 9))
ax1.set_title('PCA')
print("Calculated PCA successfully!")

# Calculating t-SNE
tsne = TSNE(n_components=2, perplexity=25, random_state=42,init='pca').fit_transform(features)
ax2 = axes[1]
ax2.scatter(tsne[:,0], tsne[:,1], c=data['Class'], cmap=plt.cm.get_cmap("jet", 9))
ax2.set_title('t-SNE')
print("Calculated t-SNE successfully!")

# Calculating UMAP
reducer = umap.UMAP()
embedding = reducer.fit_transform(features)
ax3 = axes[2]
ax3.scatter(embedding[:,0], embedding[:,1], c=data['Class'], cmap=plt.cm.get_cmap("jet", 9))
ax3.set_title('UMAP')
print("Calculated UMAP successfully!")

# Plotting all three together
handles = [plt.scatter([],[],color=plt.cm.get_cmap("jet", 9)(i/9), label=classes[i]) for i in range(9)]
fig = plt.gcf()
# legend on the center right by the side of all three plots
fig.legend(handles=handles, loc='center right', bbox_to_anchor=(0.97, 0.5), ncol=1)
# displace the plots a bit to the left
fig.subplots_adjust(right=0.85)

fig.suptitle('Visualization of the Malware dataset', fontsize=16)
plt.show()
