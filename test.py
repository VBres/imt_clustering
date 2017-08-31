import imt_clustering_pkg as imt_clus
G = imt_clus.IMTGraph.read_csv('data/facebook_combined.txt', delimiter =' ')
C = imt_clus.AgglomerativeHierarchical_ocl(G)
C.fit()
C.save_csv()