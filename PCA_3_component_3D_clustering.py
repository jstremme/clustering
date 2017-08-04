import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
from collections import Counter
from scipy import cluster
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def plot_pca(X):

	pca = PCA().fit(X)
	plt.plot(np.cumsum(pca.explained_variance_ratio_))
	plt.xlabel('Number of Componenets')
	plt.ylabel('Cumulative Explained Variance')
	plt.title('PCA')
	plt.show()

def plot_variance_explained(initial):

	plt.plot([var for (cent,var) in initial])
	plt.xlabel('Num Clusters K', fontsize=14)
	plt.ylabel('Unexplained Variance', fontsize=14)
	plt.title('Variance as a function of Clusters K')
	plt.show()

def scatter_points(X, ax, colors, labels, dependent_vars, mark_dependent_vars):

	f = open("coordinate_log_pca.txt", "w")

	for i in range(len(X)):

		marker = None
		if mark_dependent_vars:
			if dependent_vars[i] == 1:
				marker = '+'
			elif dependent_vars[i] == 0:
				marker = '_'
			else:
				marker = None

		ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker=marker)

		f.write("coordinate: {}, label: {}, color: {}, marker: {}"\
	 	 .format(X[i], labels[i], colors[labels[i]], marker)+"\n")

	f.close()

def plot_clusters(X, num_clusters, dependent_vars, mark_dependent_vars):

	kmeans = KMeans(n_clusters=num_clusters)
	kmeans.fit(X)

	centroids = kmeans.cluster_centers_
	labels = kmeans.labels_

	print()
	print("Centroids: ")
	print()
	print(centroids)
	print()

	color_options = ["g", "r", "b", "y", "c", "m", "b", "w"]
	colors = color_options[0:num_clusters]

	c = Counter(labels)
	fig = figure()
	ax = fig.gca(projection='3d')

	scatter_points(X, ax, colors, labels, dependent_vars, mark_dependent_vars)

	for num_clusters in range(num_clusters):
		print("Cluster {} contains {} samples".format(num_clusters, c[num_clusters]))

	print()

	ax.scatter(centroids[:, 0],centroids[:, 1], centroids[:, 2],\
			 marker = "x", s=150, linewidths = 5, zorder = 100, c=colors)

	plt.title('K Means Clusters')
	plt.show()

def main():

	input_data = np.array([[1, 2, 5, 7, 0],
				[5, 8, 2, 9, 1],
				[1.5, 1.8, 6, 8, 0],
				[8, 8, 9, 6, 1],
				[1, 0.6, 10, 15, 1],
				[2.5, 3.8, 6, 3, 0],
				[2.5, 5.8, 9, 5, 1],
				[5, 8, 3, 4, 1],
				[4, 0.6, 7, 3, 0],
				[2.5, 1.8, 4.6, 2.2, 0],
				[6.5, 1.8, 12, 10.1, 1],
				[7, 8, 9, 9, 1],
				[2, 0.6, 7, 5, 0],
				[5.5, 1.8, 4, 3, 0],
				[4.8, 6.9, 6, 5, 0],
				[4.9, 9.8, 2, 6, 1],
				[9, 11, 12, 11, 1]])

	dependent_vars = list(input_data[:,-1])
	independent_vars = input_data[:,0:input_data.shape[1]-1]

	X_std = StandardScaler().fit_transform(independent_vars)
	plot_pca(X_std)

	pca = PCA(n_components=3)
	X_PCA = pca.fit_transform(X_std)
	X = X_PCA

	initial = [cluster.vq.kmeans(X,i) for i in range(1,10)]
	plot_variance_explained(initial)
	
	num_clusters = int(input("Enter desired number of clusters as an integer: "))
	mark_dependent_vars = str(input("Classify points by dependent variable (yes or no): ")).lower()

	if mark_dependent_vars == "yes":
		mark_dependent_vars = True
	elif mark_dependent_vars == "no":
		mark_dependent_vars = False
	else:
		raise ValueError("--Expected yes or no for classify dependent variable option--")

	plot_clusters(X, num_clusters, dependent_vars, mark_dependent_vars)

if __name__ == '__main__':

	plt.style.use('ggplot')
	main()