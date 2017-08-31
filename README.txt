A graph clustering package, using GPU parallelization to work on larger graphs.

Requirements:
PyOpenCL
NetworkX (optionnal)


Example of use:

import imt_clustering_pkg as imt_clus
G=imt_clus.IMTGraph.read_csv(filepath, delimiter=' ', columns = (0,1), skipline=0)
C=imt_clus.RegionsGrowing_ocl(G, n_clusters=5, seed=42, itermax=1000)
C.fit()
C.save_csv(filpath='clusters.csv', delimiter=' ')

some graphs can be tested in data/

API:

class imt_clustering_pkg.IMTGraph(names=None, n=None, m=None, neighbor_table=None, neighbor_id=None)
Describe a graph. Structure used for parrallel clustering. Currently, only undirected, unweighted graph are available.
Nodes (vertices) are indexed with integers from 0 to (number_of_nodes-1)
Attributes:
	nodes_names: a list or array of n elements. To use a different labelling of the graph. names[i] contains the name of node i
	nodes_number: int, number of nodes in the graph.
	edges_number: int, total number of vertices in the graph (undirected edges count twice).
	_neighbor_table: list or array of int (edges_number elements), contain the neighors list of each node.
	_neighbor_id : list or array of int (nodes_number+1 elements), contain the info needed to get the neighborhood of each node

Methods:
	read_csv(filepath, delimiter=',', columns = (0,1), skipline = 0)
	read a graph from a csv file containing all the edges of the graph
		filepath: string, full path of the graph to read
		delimiter: string, optional(default=','), character used for column delimitation
		columns: tuple of int, optional(default=(0,1)), columns to be used in the read file
		skipline: int, optional(default=0), number of lines to skip at the begining of the file

	to_nx()
	transform an IMTGraph into its NetworkX equivalent

	neighbors(index)
	returns the list of the neighbors of a given node
		index: int, index of the node

class imt_clus.RegionsGrowing_ocl(G, n_clusters=5, seed=None, itermax=1000)
The class used to perform a region growing clustering on a given graph.
Parameters:
	G: class imt_clustering_pkg.IMTGraph, the graph to be clusterised
	n_clusters: int, optional (default=5), number of clusters searched
	seed: int or list of int, optinal (default=None), if seed is a list, the list or the starting points is this list. Else, a random number generator is fixed with the given int seed. If None is given, the RNG is seeded with the current time
	itermax: int, optional(default=1000), number max of iterations
Attributes:
	G, IMTGraph
	nodes_seed: list of int, initial nodes
	n_clusters: int, number of clusters
	itermax: int, maximum number of iteration
	clusters: list of int, clusters[i] contains the cluster number of node i
Methods:
	fit()
	Asks for the device to use, then performs the clustering. Print informations about the clustering.
	
	save_csv(path='clusters.csv', delim=',')
	Save the clustering under a csv file with two columns "node" and "cluster"
		path: string, optional(default='clusters.csv') full path of the file
		delim: string, optional(default=','), separation character to use

class imt_clus.RegionsGrowing_nx(G, n_clusters=5, seed=None, itermax=1000)
The class used to perform a region growing clustering on a given graph. (uses NetworkX)
Parameters:
	G: NetworkX.Graph, the graph to be clusterised
	n_clusters: int, optional (default=5), number of clusters searched
	seed: int or list of int, optinal (default=None), if seed is a list, the list or the starting points is this list. Else, a random number generator is fixed with the given int seed. If None is given, the RNG is seeded with the current time
	itermax: int, optional(default=1000), number max of iterations
Attributes:
	G, NetworkX.Graph
	nodes_seed: list of int, initial nodes
	n_clusters: int, number of clusters
	itermax: int, maximum number of iteration
	clusters: list of int, clusters[i] contains the cluster number of node i
Methods:
	fit()
	Performs the clustering. Print informations about the clustering.
	
	save_csv(path='clusters.csv', delim=',')
	Save the clustering under a csv file with two columns "node" and "cluster"
		path: string, optional(default='clusters.csv') full path of the file
		delim: string, optional(default=','), separation character to use


class imt_clus.AgglomerativeHierarchical_ocl(G, n_proc=10, n_added=10)
The class used to perform an agglomerative hierarchical clustering on a given graph. (uses NetworkX)
Parameters:
	G: class imt_clustering_pkg.IMTGraph, the graph to be clusterised
	n_proc: int, optional (default=10), the number of pairs considered at each step. Will be always < to the maximum number of processors available at the same time. If called with n_proc="max", n_proc will be set to the maximum number of processors available at the same time.
	n_added: int, optional (default=10), the maximum number of merging done at each step.
Attributes:
	G, imt_clustering_pkg.IMTGraph
	n_proc: int
	n_added: int 
	clusters: list list of int, clusters[i] contains the list of nodes in cluster i
Methods:
	fit()
	Asks for the device to use, then performs the clustering. Print informations about the clustering.
	
	save_csv(path='clusters.csv', delim=',')
	Save the clustering under a csv file with two columns "node" and "cluster"
		path: string, optional(default='clusters.csv') full path of the file
		delim: string, optional(default=','), separation character to use