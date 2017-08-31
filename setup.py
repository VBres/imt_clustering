from setuptools import setup

setup(name='imt_clustering_pkg',
      version='0.1',
      description='A graph clustering package',
      packages=['imt_clustering_pkg'],
      install_requires=[
          'networkx',
		  'pyopencl',
      ],
      zip_safe=False)