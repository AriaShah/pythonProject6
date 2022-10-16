import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()

df = pd.read_csv('C:/Users/HP/Desktop/ir.csv')
df.dropna(axis=0,how='any',subset=['population'],inplace=True)


# Variable with the Longitude and Latitude
X_weighted =df.loc[:,['city','lat','lng','population']]

X_weighted.head(3)

lat_long = X_weighted[X_weighted.columns[1:3]]
pop_size = X_weighted[X_weighted.columns[3]]
sample_weight = pop_size

# kmeans with k=1
K_clusters = 1
kmeans = KMeans(n_clusters=K_clusters)
kmeans.fit(lat_long, sample_weight = pop_size)
SSE = [kmeans.inertia_]

# Finding the optimal k
while True:
    K_clusters+=1
    kmeans = KMeans(n_clusters=K_clusters)
    kmeans.fit(lat_long, sample_weight = pop_size)
    SSE.append(kmeans.inertia_)
    if SSE[K_clusters-1]/SSE[0] <= 0.05:
        break

# Elbow plot
Kplot = [i for i in range(1,K_clusters+1)]
plt.plot(Kplot, SSE)
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.title('Elbow Curve - Weighted')
plt.show()


# Final kmeans with optimal k
kmeans = KMeans(n_clusters = K_clusters, max_iter=1000, init ='k-means++')

lat_long = X_weighted[X_weighted.columns[1:3]]
pop_size = X_weighted[X_weighted.columns[3]]
weighted_kmeans_clusters = kmeans.fit(lat_long, sample_weight = pop_size) # Compute k-means clustering.
X_weighted['cluster_label'] = kmeans.predict(lat_long, sample_weight = pop_size)

centers = kmeans.cluster_centers_ # Coordinates of cluster centers.
assign = kmeans.labels_
labels = X_weighted['cluster_label'] # Labels of each point

X_weighted.plot.scatter(x = 'lat', y = 'lng', c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=300, alpha=0.5)
plt.title('Clustering GPS Co-ordinates to Form Regions - Weighted',fontsize=18, fontweight='bold')
plt.show()