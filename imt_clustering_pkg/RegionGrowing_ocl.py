import pyopencl as cl
import numpy as np
import csv
import random
import sys
# from IMTGraph import IMTGraph
from time import time

class RegionGrowing_ocl:
	def __init__(self, G, n_clusters=5, seed=None, itermax=1000):
		self.G = G
		n_nodes = self.G.nodes_number
		if type(seed)==int:
			random.seed(seed)
			self.nodes_seed = random.sample(range(n_nodes),n_clusters)
			self.n_clusters = n_clusters
		elif type(seed)==list:
			self.n_clusters = len(seed)
			self.nodes_seed = seed
		else:
			random.seed(time())
			self.nodes_seed = random.sample(range(n_nodes),n_clusters)
			self.n_clusters = n_clusters			
		self.itermax = itermax
		self.clusters = np.array([-1 for x in range(n_nodes)], np.int32)
	
		for i in range(self.n_clusters):
			self.clusters[self.nodes_seed[i]]=i

	def fit(self):
		n_nodes = self.G.nodes_number

		# Set up OpenCL
		context = cl.create_some_context()
		queue = cl.CommandQueue(context)
		
		# Create the compute program from the source buffer and build it
		kernelsource = """
		__kernel void grow(
			const long n_nodes, 
			const int n_clusters, 
			__global long* neighb_table, //vector containing all the edges of the graph
			__global long* neighb_id, //neighb_table[neighb_id[i]] contains 1 neighbor of node i
			__global int* clusters_in, //clusters[i]=k means node i is in cluster k (initially -1)
			__global int* clusters_out,
			__global int* clusters_priority)
		{
			int i= get_global_id(0);

			if (i < n_nodes && clusters_in[i]==-1) { //process only nodes that aren't already in a cluster
				long id = neighb_id[i];
				int n_neighbors = neighb_id[i+1]-id;
				int k = -1;
				for(int j=id; j< id + n_neighbors; j++){
					if(k==-1){
						k=clusters_in[neighb_table[j]];
					}
					else if(clusters_in[neighb_table[j]]!=-1 && clusters_priority[clusters_in[neighb_table[j]]] < clusters_priority[k]){
						k=clusters_in[neighb_table[j]];
					}
				}
				
				if(k!=-1){
					clusters_out[i] = k;
				}
			}
		}



		__kernel void count_cluster(
			const long n_nodes,
			const int n_clusters,
			__global int* clusters,
			__global int* clusters_size)
		{	
			int k = get_global_id(0);
			for (int i=0; i<=n_clusters; i++){
				clusters_size[4*i+k]=0;
			}
			for(int i=k*n_nodes/4; i<(k+1)*n_nodes/4; i++){
				if (clusters[i]>=0){
					clusters_size[4*clusters[i]+k]+=1;
				}
			}
		}
		"""
		program = cl.Program(context, kernelsource).build()
		grow = program.grow
		grow.set_scalar_arg_dtypes([np.int64, np.int32, None, None, None, None, None])
		count_cluster = program.count_cluster
		count_cluster.set_scalar_arg_dtypes([np.int64, np.int32, None, None])
		
		# Create OpenCL buffers
		neighb_table = np.array(self.G._neighbor_table, np.int64)
		neighb_table_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=neighb_table)
		
		neighb_id = np.array(self.G._neighbor_id, np.int64)
		neighb_id_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=neighb_id)
		
		clusters_in_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.clusters)
		clusters_out_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.clusters)
		
		clusters_sizes = np.array([[1]*4]*self.n_clusters, np.int32)
		clusters_size_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=clusters_sizes)
		
		clusters_priority = np.array([1]*self.n_clusters, np.int32)
		clusters_priority_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=clusters_priority)
		
		nbvisited = 0
		nbiter = 0
		nbvisited_last = -1
		while nbiter<self.itermax and nbvisited<n_nodes and nbvisited_last<nbvisited:
			grow(queue, (n_nodes,), None , n_nodes, self.n_clusters,
			neighb_table_buf,
			neighb_id_buf,
			clusters_in_buf,
			clusters_out_buf,
			clusters_priority_buf)
			
			queue.finish()
			
			# Update input
			cl.enqueue_copy(queue, clusters_in_buf, clusters_out_buf)
			
			if (nbiter%100)==0 or nbiter==self.itermax:
				# Once in a while check if the clustering is over
				count_cluster(queue, (4,), None, n_nodes, self.n_clusters, clusters_in_buf, clusters_size_buf)
				queue.finish()
				
				cl.enqueue_copy(queue, clusters_sizes, clusters_size_buf)
				clusters_size = np.sum(clusters_sizes, axis=1)
				nbvisited_last = nbvisited
				nbvisited = clusters_size.sum()
				clusters_priority = np.array(clusters_size, np.int32)
				cl.enqueue_copy(queue, clusters_priority_buf, clusters_priority)
			nbiter += 1
			
		cl.enqueue_copy(queue, self.clusters, clusters_out_buf)
		
		# Compute modularity Q
		E = np.zeros((self.n_clusters, self.n_clusters))
		for node1 in range(n_nodes):
			i = self.clusters[node1]
			for node2 in self.G.neighbors(node1):
				j = self.clusters[node2]
				E[i,j]+=1
				E[j,i]+=1
		E = E/(2*self.G.edges_number)
		
		Q = 0
		for i in range(self.n_clusters):
			a = 0
			for j in range(self.n_clusters):
				a += E[i,j]
			Q+= E[i,i]-a**2
		self.Q = Q
		
		print("(" + str(self.n_clusters)+" clusters)")
		print(str(nbvisited)+" nodes visited out of "+ str(n_nodes)+" in "+str(nbiter)+" iterations")
		print("Mean cluster size: "+str(clusters_size.mean()))
		print("Standard deviation: "+str(clusters_size.std()))
		print("Modularity: "+str(Q))
	
	def save_csv(self, path='clusters.csv', delim=','):
		with open(path,'w' )as csvfile:
			writer = csv.writer(csvfile, delimiter=delim)
			writer.writerow(['node','cluster'])
			for i in range(self.G.nodes_number):
				writer.writerow([self.G.nodes_names[i],self.clusters[i]])


if __name__ == '__main__':
	filepath = '../data/facebook_combined.txt'
	seed = 42
	k = 5
	delim=' '

	if len(sys.argv)>=2 :
		filepath = sys.argv[1]
	if len(sys.argv)>=3:
		seed = int(sys.argv[2])
	if len(sys.argv)>=4:
		k = int(sys.argv[3])
	if len(sys.argv)>=5:
		print('Usage: python RegionsGrowing_ocl filepath seed number_of_clusters')
		sys.exit()
	time0 = time()
	G = IMTGraph.read_csv(filepath, delim)
	print("File loaded in "+str(time()-time0)+"s")
	C = RegionGrowing_ocl(G, k, seed, 10000)
	C.fit()
	C.save_csv()

