"""
	Python package used for the clustering of larger graphs
	
	Example of use:
	
	import imt_clustering_pkg as imt_clus
	G = imt_clus.IMTGraph.read_csv('data/facebook_combined.txt', delimiter= ' ')
	C = imt_clus.RegionGrowing_ocl(G)
	C.fit()
	C.save_csv('clusters.csv', delimiter= ' ')
"""

from imt_clustering_pkg.IMTGraph import IMTGraph
from imt_clustering_pkg.RegionGrowing_nx import RegionGrowing_nx
from imt_clustering_pkg.RegionGrowing_ocl import RegionGrowing_ocl
from imt_clustering_pkg.AgglomerativeHierarchical_ocl import AgglomerativeHierarchical_ocl
