import csv
import numpy as np
try:
	import networkx as nx
except ImportError:
	print('Warning: NetworkX not available')

def ComputeRanksInList(inputList):
  """
  Compute the rank of the elements in the list inputList. Returns a list 
  ListRanks of the same size as inputList with the rank of each element 
  in inputList. Note that identical elements in inputList will have the
  same rank.
  
  Algorithmic cost is n*(2+log(n)). Tested with lists containing strings of size 20 and numpy arrays with floats, leading to 
  a similar computational burden:
  -> about 11Go and 8 minutes if len(inputList)=100000000
  -> about 1Go and 30 seconds if len(inputList)=10000000
  -> about          2 seconds if len(inputList)=1000000
  -> about        0.1 seconds if len(inputList)=100000
  """
  
  #1) first sort (two elements of inputList with the same name won't have the same rank so a post-treatment will be necessary)
  ListRanks = [0] * len(inputList)
  for i, x in enumerate(sorted(range(len(inputList)), key=lambda y: inputList[y])):
    ListRanks[x] = i
  
  #2) generate the list NodeNames where the index of the name is its rank (with redundancy for the reasons explained above)
  NodeNames=['NoName']*len(inputList)
  for i in range(len(inputList)):
    NodeNames[ListRanks[i]]=inputList[i]
  
  #3) post-treatment so that redundant elements of inputList have a single reference index
  NodeNamesID=list(range(len(NodeNames)))
  
  LastNodeName=NodeNames[0]
  CurrentNewIndex=0
  NodeNamesID[0]=CurrentNewIndex
  
  for i in range(1,len(inputList)):
    if LastNodeName!=NodeNames[i]:
      LastNodeName=NodeNames[i]
      CurrentNewIndex+=1
    NodeNamesID[i]=CurrentNewIndex
    
  #print NodeNames
  #print NodeNamesID
  
  del NodeNames
    
  #4) modify the indices of ListRanks
  for i in range(len(inputList)):
    ListRanks[i]=NodeNamesID[ListRanks[i]]
  
  del NodeNamesID
  
  return ListRanks

def ListSort(inputList,ListRanks):
  """
  return a list in which the elements of inputList are ranked with the ranks of ListRanks.
  -> ListRanks must have the same size as inputList and its ranks should start at 0
  -> ListRanks can contain redundant ranks
  -> ListRanks has ideally no gap between 0 and its maximal value
  """
  NodeNames=['NoName']*(max(ListRanks)+1)
  for i in range(len(inputList)):
    NodeNames[ListRanks[i]]=inputList[i]
   
  return NodeNames
 
class IMTGraph:
"""
Describe a graph. Currently, only undirected, unweighted graph are available.
Nodes (vertices) are labelled with integers from 0 to (number_of_nodes-1)
"""
	
	def __init__(self):
		self.nodes_names = []
		self.nodes_number = 0
		self.edges_number = 0
		self._neighbor_table = []
		self._neighbor_id = []
		self._edges_weights = None
		
		
	def __init__(self, names, n, m, neighbor_table, neighbor_id, weights = None):
		self.nodes_names = names
		self.nodes_number = n
		self.edges_number = m
		self._neighbor_table = neighbor_table
		self._neighbor_id = neighbor_id
		self._edges_weights = weights
	
	@classmethod
	def read_csv(cls, filepath, delimiter=',', columns = (0,1), skipline = 0, weight_column = None):
		"""
		Read an undirected graph from a csv file containing the edges of the graph and returns the created object.
		The file is expected to have at least 2 columns, each row representing an edge
		If there is a Header, specify the number skipline to the number of lines to be skipped
		You can specify the column numbers containing the edges if the file has additionnal columns
		If the graph is weighted, specify the column number containing the weights in weight_column (not implemented)
		"""

		if len(columns)!=2:
			raise InputError('Please provide a tuple of 2 column numbers')
		if weight_column == None:
			weights = None
			weighted = False
		else:
			weights = []
			weighted = True
		from_node = []
		to_node = []
		
		with open(filepath,'r') as csvfile:
			reader = csv.reader(csvfile, delimiter=delimiter, quotechar='%')
			# remove header
			for i in range(skipline):
				row = next(reader, None)
			# read data
			for row in reader:
				from_node.append(row[columns[0]])
				to_node.append(row[columns[1]])
				if weighted:
					weights.append(row[weight_column])
		
		input_list = from_node + to_node
		list_ranks = ComputeRanksInList(input_list)
		nodes_names = ListSort(input_list,list_ranks)
		nodes_number = len(nodes_names)
		edges_number = len(from_node)
		neighborhood_list = [set() for dummy in range(nodes_number)]
		for i in range(edges_number):
			node_i = list_ranks[i]
			node_j = list_ranks[i+edges_number]
			neighborhood_list[node_i].add(node_j)
			neighborhood_list[node_j].add(node_i)
		# edges_number = 0
		neighbor_size = [len(x) for x in neighborhood_list]
		edges_number = np.sum(neighbor_size)
		neighbor_id = [0]
		for i in range(nodes_number):
			neighbor_id.append(neighbor_id[i]+neighbor_size[i])
		neighbor_table = [node for neighborhood in neighborhood_list for node in neighborhood]
		
		print("The file "+filepath+" was loaded and contains "+str(nodes_number)+" nodes and "+str(edges_number)+" edges.")
		
		return(cls(nodes_names, nodes_number, edges_number, neighbor_table, neighbor_id))
		
	def neighbors(self, index):
		"""
		Returns the list containing the neighbors of the node with the given index
		"""
		try:
			return(self._neighbor_table[self._neighbor_id[index]:self._neighbor_id[index+1]])
		except IndexError:
			if index == (self.nodes_number-1):
				return(self._neighbor_table[self._neighbor_id[index]:])
			else:
				raise IndexError('Node index must be < ' + str(self.nodes_number))
				
	def to_nx(self):
		"""
		Convert the Graph to NetworkX format
		"""
		G=nx.Graph()
		for i in range(self.nodes_number):
			for j in range(self._neighbor_id[i],self._neighbor_id[i+1]):
				G.add_edge(self.nodes_names[i], self.nodes_names[self._neighbor_table[j]])
		return(G)
	

if __name__ == '__main__':
	G = IMTGraph.read_csv('../data/facebook_combined.txt', ' ')
