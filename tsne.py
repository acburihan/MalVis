import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Importing features.csv and trainLabels.csv
features = pd.read_csv('features.csv')
data = pd.read_csv('trainLabels.csv')

tsne = TSNE().fit_transform(features)

plt.scatter(tsne[:,0], tsne[:,1], c=data['Class'], cmap=plt.cm.get_cmap("jet", 9))
plt.colorbar(ticks=range(10))
plt.show()