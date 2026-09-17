from sklearn.datasets import make_moons
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
X1, label2 = make_moons(n_samples=200, noise=0.08, random_state=123)
plt.scatter(X1[:, 0], X1[:, 1], c=label2, alpha=0.7)


agglo = AgglomerativeClustering(n_clusters=2)
agglo.fit(X1)
myColors={0:'red', 1:'green'}
plt.scatter(X1[:, 0], X1[:, 1], c=pd.Series(agglo.labels_).apply(lambda x: myColors[x]), alpha=0.7)
plt.title('Dataset #1: Agglomerative')
plt.show()