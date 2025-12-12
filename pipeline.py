import pandas as pd

## import generated features

features = pd.read_csv('features_output_1.csv.gz', compression='gzip')
labels = pd.read_csv('data/40mhz/original_labels/brc-2002_086400-01-output_true_labels.csv.gz', compression='gzip')

## implement kmeans clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(features)
clusters = kmeans.labels_
features['cluster'] = clusters

## compare clusters with true labels
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(labels['class'], features['cluster'])
print(f'Adjusted Rand Index between KMeans clusters and true labels: {ari}')

