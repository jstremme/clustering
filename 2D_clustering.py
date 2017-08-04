import numpy as np
from scipy import cluster
from matplotlib import pyplot

def plot_variance_explained(initial):

	pyplot.plot([var for (cent,var) in initial])
	pyplot.xlabel('Num Clusters K', fontsize=14)
	pyplot.ylabel('Unexplained Variance', fontsize=14)
	pyplot.title('Variance as a function of Clusters K')
	pyplot.show()

def plot_clusters(input_data, initial, num_clusters):

	cent, var = initial[num_clusters-1]
	print()
	print(cent)
	print()

	assignment, cdist = cluster.vq.vq(input_data,cent)
	pyplot.scatter(input_data[:,0], input_data[:,1], c=assignment)
	pyplot.title('K Means Clusters')
	pyplot.show()

def main():

	np.random.seed(123)
	input_data = np.reshape(np.random.uniform(0,100,60), (30,2))
	
	initial = [cluster.vq.kmeans(input_data,i) for i in range(1,10)]
	plot_variance_explained(initial)

	print()
	num_clusters = int(input("Enter desired number of clusters as an integer: "))
	plot_clusters(input_data, initial, num_clusters)

if __name__ == '__main__':

	pyplot.style.use('ggplot')
	main()
