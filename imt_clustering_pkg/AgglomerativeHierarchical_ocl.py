import pyopencl as cl
import numpy as np
import csv
import sys
from time import time

def get_neighbors(idx, neighb_id, neighb_table, edges_weights):
	try: 
		neighbors = neighb_table[neighb_id[idx]:neighb_id[idx+1]]
		edges_weights = edges_weights[neighb_id[idx]:neighb_id[idx+1]]
	except IndexError:
		neighbors = neighb_table[neighb_id[idx]:]
		edges_weights = edges_weights[neighb_id[idx]:]
	return neighbors,edges_weights

def custom_index(l, obj):
	try:
		return l.index(obj)
	except ValueError:
		return -1

def remove_all(l, obj):
	while 1:
		try:
			l.remove(obj)
		except ValueError:
			return
			
class AgglomerativeHierarchical_ocl:

	def __init__(self, G, n_proc=10, n_added=10):
		self.G = G
		self.n_proc = n_proc
		self.n_added=n_added

			
	def fit(self):
		start_time = time()
		N = self.G.nodes_number
		M = self.G.edges_number
		
		# Set up OpenCL
		context = cl.create_some_context()
		queue = cl.CommandQueue(context)

		
		# Create the compute program from the source buffer and build it
		kernelsource = """
		long index(long* list, long element)
		{
			long i=0;
			while(list[i]!=element){
				i++;
			}
			return(i);
		}

		__kernel void find_neighbor(
			const long n_nodes, 
			__global long* neighb_table, //vector containing all the edges of the graph
			__global long* neighb_id, //neighb_table[neighb_id[i]] contains 1 neighbor of node i
			__global float* edges_weights,
			__global float* nodes_weights,
			__global float* nodes_here,
			__global long* selected_nodes,
			__global float* deltaQ,
			__global long* best_neighb)
		{
			long work_id = get_global_id(0);
			long i = selected_nodes[work_id];
			float ai = nodes_weights[i];
			float deltaQ_tmp = -1.0;
			long k = -1;
			long id = neighb_id[i];
			long id2 = neighb_id[i+1];
			
			for(int j=id; j< id2; j++) {
				float tmp =  2*(edges_weights[j] - nodes_weights[index(nodes_here, neighb_table[j])]*ai);
				if( tmp > deltaQ_tmp){
					deltaQ_tmp = tmp;
					k = neighb_table[j];
				}
			}
			best_neighb[work_id] = k;
			deltaQ[work_id] = deltaQ_tmp;
		}
		"""
		
		# Initialisation
		program = cl.Program(context, kernelsource).build()
		find_neighbor = program.find_neighbor
		find_neighbor.set_scalar_arg_dtypes([np.int64, None, None, None, None, None, None, None, None])
		device = context.devices[0]
		nwork_groups = device.max_compute_units
		work_group_size = find_neighbor.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, device)
		n_max = nwork_groups*work_group_size
		if self.n_proc == "max" or self.n_proc>n_max:
			self.n_proc = n_max
		
		neighb_table = self.G._neighbor_table
		neighb_id = self.G._neighbor_id
		edges_weights = [1/M for i in range(M)]
		nodes_weights = [len(self.G.neighbors(i))/M for i in range(N)]
		nodes_here = list(range(N))
		
		Q = 0
		for a in nodes_weights:
			Q = Q-a**2
		best_it = 0
		maxQ = Q
		ocl_time = 0
		ocl_func_time = 0
		ocl_buff_time = 0
		copy_time = 0
		merge_list = []
		
		# Perform clustering
		while N>1:
			time0 = time()
			p = min(self.n_proc, N)
			neighb_table_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.asarray(neighb_table, np.int64))
			neighb_id_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.asarray(neighb_id, np.int64))
			edges_weights_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.asarray(edges_weights, np.float32))
			nodes_weights_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.asarray(nodes_weights, np.float32))
			nodes_here_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.asarray((nodes_here), np.int64))
			deltaQ = np.empty(p).astype(np.float32)
			deltaQ_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, deltaQ.nbytes)
			best_neighbor = np.empty(p).astype(np.int64)
			best_neighbor_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, best_neighbor.nbytes)
			
			selected_nodes = np.argsort(nodes_weights)[0:p]
			selected_nodes_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=selected_nodes)
			time15 = time()
			ocl_buff_time += time15 - time0
			
			find_neighbor(queue, (p,), None, N,
			neighb_table_buf,
			neighb_id_buf,
			edges_weights_buf,
			nodes_weights_buf,
			nodes_here_buf,
			selected_nodes_buf,
			deltaQ_buf,
			best_neighbor_buf)
			
			queue.finish()
			# Read output
			cl.enqueue_copy(queue, deltaQ, deltaQ_buf)
			cl.enqueue_copy(queue, best_neighbor, best_neighbor_buf)
			time1 = time()
			ocl_time += time1-time0
			ocl_func_time += time1 - time15
			
			candidates = np.argsort(deltaQ)[::-1]
			selected = set()
			i_list = []
			j_list = []
			for k in range(min(p,self.n_added)):
				i = min(nodes_here[selected_nodes[candidates[k]]],best_neighbor[candidates[k]])
				j = max(nodes_here[selected_nodes[candidates[k]]],best_neighbor[candidates[k]])
				if i not in selected and j not in selected:
					selected.add(i)
					selected.add(j)
					i_list.append(i)
					j_list.append(j)
					Q = Q + deltaQ[candidates[k]]

			merge_list+= [(i_list[k], j_list[k]) for k in range(len(i_list))]		
			
			if Q>maxQ:
				maxQ = Q
				best_it = len(merge_list)
			

			time2 = time()
			new_neighb_id = [0]
			new_neighb_table = []
			new_edges_weights = []
			new_nodes_weights = []
			# copy the modified graph
			for k in range(N):
				pair_idx = custom_index(j_list, nodes_here[k])
				if pair_idx==-1: #node_here[k] isn't a node_j to be suppressed
					neighbors, new_edges_weights_tmp = get_neighbors(k, neighb_id, neighb_table, edges_weights)
					new_nodes_weights.append(nodes_weights[k])
					for node_j in j_list: #change all edges going to node_j 
						j_idx_in_neigh = custom_index(neighbors, node_j)
						if j_idx_in_neigh!=-1:
							pair_idx = j_list.index(node_j)
							node_i = i_list[pair_idx]
							i_idx_in_neigh = custom_index(neighbors, node_i)
							if i_idx_in_neigh==-1:
								neighbors[j_idx_in_neigh]=node_i
							else:
								new_edges_weights_tmp[i_idx_in_neigh]+=new_edges_weights_tmp[j_idx_in_neigh]
								del new_edges_weights_tmp[j_idx_in_neigh]
								del neighbors[j_idx_in_neigh]
							
					pair_idx = custom_index(i_list, nodes_here[k])
					if pair_idx!=-1: #node_here[k] is a node_i to be updated
						node_i = nodes_here[k]
						node_j = j_list[pair_idx]
						j_idx = nodes_here.index(node_j)
						neighbors_j,edges_weights_j = get_neighbors(j_idx, neighb_id, neighb_table, edges_weights)
						i_idx_in_neigh = custom_index(neighbors, node_i)
						if i_idx_in_neigh!=-1:
							del neighbors[i_idx_in_neigh]
							del new_edges_weights_tmp[i_idx_in_neigh]
						i_idx_in_neigh = custom_index(neighbors_j, node_i)
						if i_idx_in_neigh!=-1:
							del neighbors_j[i_idx_in_neigh]
							del edges_weights_j[i_idx_in_neigh]					
						for node_j2 in j_list: #change all edges going to node_j2
							j_idx_in_neigh = custom_index(neighbors_j, node_j2)
							if j_idx_in_neigh!=-1:
								pair_idx = j_list.index(node_j2)
								node_i = i_list[pair_idx]
								i_idx_in_neigh = custom_index(neighbors_j, node_i)
								if i_idx_in_neigh==-1:
									neighbors_j[j_idx_in_neigh]=node_i
								else:
									edges_weights_j[i_idx_in_neigh]+=edges_weights[j_idx_in_neigh]
									del edges_weights_j[j_idx_in_neigh]
									del neighbors_j[j_idx_in_neigh]
						for k2,neighb_j in enumerate(neighbors_j):
							neighb_j_idx = custom_index(neighbors, neighb_j)
							if neighb_j_idx != -1:
								new_edges_weights_tmp[neighb_j_idx]+= edges_weights_j[k2]
							else:
								neighbors.append(neighb_j)
								new_edges_weights_tmp.append(edges_weights_j[k2])
						new_nodes_weights[-1]+= nodes_weights[j_idx]
					
					new_neighb_table += neighbors
					new_edges_weights += new_edges_weights_tmp
					new_neighb_id.append(new_neighb_id[-1]+len(neighbors))
			for node_j in j_list:
				nodes_here.remove(node_j)
			N = len(nodes_here)
			
			neighb_table = new_neighb_table
			neighb_id = new_neighb_id
			edges_weights = new_edges_weights
			nodes_weights = new_nodes_weights
			copy_time += time()-time2
		end_time = time()
		clusters = []
		for k in range(1,best_it+1):
			a,b = merge_list[best_it-k]
			i = 0
			while i<len(clusters) and a not in clusters[i]:
					i+=1
			if i>=len(clusters):
				clusters.append({a,b})
			else:
				clusters[i].add(b)
		self.clusters=clusters
		n_clusters= len(clusters)
		clusters_size = np.array([len(x) for x in clusters])
		print("(" + str(n_clusters)+" clusters)")
		print('maxQ: ' + str(maxQ))
		print("Mean cluster size: "+str(clusters_size.mean()))
		print("Standard deviation: "+str(clusters_size.std()))
		
	def save_csv(self, path='clusters.csv', delim=','):
		n_clusters = len(self.clusters)
		with open('clusters.csv','w' )as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			writer.writerow(['node','cluster'])
			for i in range(n_clusters):
				for node in self.clusters[i]:
					writer.writerow([self.G.nodes_names[node],i])
