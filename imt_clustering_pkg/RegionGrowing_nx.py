import numpy as np
import networkx as nx
import csv
import random
import sys
import math
from time import time

class RegionGrowing_nx:
	def __init__(self, G, n_clusters=5, seed=42, itermax=1000):
		self.G = G
		n_nodes = self.G.number_of_nodes()
		if type(seed)==int:
			random.seed(seed)
			self.nodes_seed = random.sample(G.nodes(),n_clusters)
			self.n_clusters = n_clusters
		elif type(seed)==list:
			self.n_clusters = len(seed)
			self.nodes_seed = seed
		else:
			print('2nd argument "seed" expects a list of starting points or an integer used for random generation')
			return()
		self.itermax = itermax

# Computing clusters
	def fit(self):
		n = self.G.number_of_nodes()
		visited = set(x for x in self.nodes_seed)
		clusters = [set([x]) for x in self.nodes_seed]
		clus_neighbors = [set(self.G.neighbors(x))-visited for x in self.nodes_seed]
		k = self.n_clusters
		nbiter =0
		nbr_visited = k
		nbr_visited_last = 0
		order = list(range(k))
		
		while nbr_visited<n and nbiter<self.itermax and nbr_visited>nbr_visited_last:
		#exit when all nodes visited or maximum number of iterations reached or no evolution in the clusters
			nbr_visited_last = nbr_visited
			new_neighbors = [set() for i in range(k)]
			random.shuffle(order)
			for i in order:
				for node in clus_neighbors[i]:
					if node not in visited:
						clusters[i].add(node)
						visited.add(node)
						new_neighbors[i] = new_neighbors[i] | set(self.G.neighbors(node))
			for i in range(k):
				clus_neighbors[i] = (clus_neighbors[i] | new_neighbors[i]) - visited
			nbiter+=1
			nbr_visited = len(visited)

		
		self.clusters = clusters
		clusters_size = np.array([len(x) for x in clusters])
		E = np.zeros((k, k))
		for i in range(k):
			for node1 in clusters[i]:
				for node2 in self.G.neighbors(node1):
					j = 0
					while not(node2 in clusters[j]):
						j +=1
					E[i,j]+=1
					E[j,i]+=1			
		E = E/(4*self.G.number_of_edges())
		
		Q = 0
		for i in range(k):
			a = 0
			for j in range(k):
				a += E[i,j]
			Q+= E[i,i]-a**2
		print("(" + str(k)+" clusters)")

		print(str(len(visited))+' nodes visited on the '+str(n)+' total')
		print("Mean cluster size: "+str(clusters_size.mean()))
		print("Standard deviation: "+str(clusters_size.std()))
		print("Modularity: "+str(Q))

	def save_csv(self, path = 'clusters.csv', delim = ' '):
			with open(path,'w' )as csvfile:
				writer = csv.writer(csvfile, delimiter=delim)
				writer.writerow(['node','cluster'])
				for i in range(self.n_clusters):
					for node in self.clusters[i]:
						writer.writerow([node,i])
	
if __name__ == '__main__':
	time0 = time()
	filepath = '../data/facebook_combined.txt'
	seed = 42
	k = 5
	delim=' '
	qtchar='%'

	# Usage: python RegionsGrowing_nx filepath seed_number k
	if len(sys.argv)>=2 :
		filepath = sys.argv[1]
	if len(sys.argv)>=3:
		seed = int(sys.argv[2])
	if len(sys.argv)>=4:
		k = int(sys.argv[3])
	if len(sys.argv)>=5:
		print('Usage: python RegionsGrowing_nx filepath seed number_of_clusters')
		sys.exit()

	# Loading the graph
	G=nx.Graph()
	with open(filepath,'r') as csvfile:
		data = csv.reader(csvfile, delimiter=delim, quotechar=qtchar)
		for row in data:
				G.add_edge(row[0],row[1])

	n = G.number_of_nodes()
	m = G.number_of_edges()
	load_time = time() - time0
	print("The file "+filepath+" was loaded in " + str(load_time)+"s and contains "+str(n)+" nodes and "+str(m)+" edges.")

	init_time = time()
	C = RegionGrowing_nx(G, k, seed, 10000)
	C.fit()
	print(time()-init_time)