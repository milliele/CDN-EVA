# -*- coding:utf-8 -*-

import random
import numpy as np
import collections
import copy, re, fnss
import networkx as nx

class DiscreteDist(object):
    """Implements a discrete distribution with finite population.

    The support must be a finite discrete set of contiguous integers
    {1, ..., N}. This definition of discrete distribution.
    """

    def __init__(self, pdf, seed=None):
        """
        Constructor

        Parameters
        ----------
        pdf : array-like
            The probability density function
        seed : any hashable type (optional)
            The seed to be used for random number generation
        """
        if np.abs(sum(pdf) - 1.0) > 0.001:
            raise ValueError('The sum of pdf values must be equal to 1')
        random.seed(seed)
        self._pdf = np.asarray(pdf)
        self._cdf = np.cumsum(self._pdf)
        # set last element of the CDF to 1.0 to avoid rounding errors
        self._cdf[-1] = 1.0

    def __len__(self):
        """Return the cardinality of the support

        Returns
        -------
        len : int
            The cardinality of the support
        """
        return len(self._pdf)

    @property
    def pdf(self):
        """
        Return the Probability Density Function (PDF)

        Returns
        -------
        pdf : Numpy array
            Array representing the probability density function of the
            distribution
        """
        return self._pdf

    @property
    def cdf(self):
        """
        Return the Cumulative Density Function (CDF)

        Returns
        -------
        cdf : Numpy array
            Array representing cdf
        """
        return self._cdf

    def rv(self):
        """Get rand value from the distribution
        """
        rv = random.random()
        # This operation performs binary search over the CDF to return the
        # random value. Worst case time complexity is O(log2(n))
        return int(np.searchsorted(self._cdf, rv) + 1)

class TruncatedZipfDist(DiscreteDist):
    """Implements a truncated Zipf distribution, i.e. a Zipf distribution with
    a finite population, which can hence take values of alpha > 0.
    """

    def __init__(self, alpha=1.0, n=1000, seed=None):
        """Constructor

        Parameters
        ----------
        alpha : float
            The value of the alpha parameter (it must be positive)
        n : int
            The size of population
        seed : any hashable type, optional
            The seed to be used for random number generation
        """
        # Validate parameters
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        if n < 0:
            raise ValueError('n must be positive')
        # This is the PDF i. e. the array that  contains the probability that
        # content i + 1 is picked
        pdf = np.arange(1.0, n + 1.0) ** -alpha
        pdf /= np.sum(pdf)
        self._alpha = alpha
        super(TruncatedZipfDist, self).__init__(pdf, seed)

    @property
    def alpha(self):
        return self._alpha

# class TruncatedNormal(DiscreteDist):
#
#     def __init__(self, a, v = 1, n=1000, seed=None):
#
#         super(TruncatedNormal, self).__init__(pdf, seed)

class Tree(collections.defaultdict):
    """Tree data structure

    This class models a tree data structure that is mainly used to store
    experiment parameters and results in a hierarchical form that makes it
    easier to search and filter data in them.
    """

    def __init__(self, data=None, **attr):
        """Constructor

        Parameters
        ----------
        data : input data
            Data from which building a tree. Types supported are Tree objects
            and dicts (or object that can be cast to trees), even nested.
        attr : additional keyworded attributes. Attributes can be trees of leaf
            values. If they're dictionaries, they will be converted to trees
        """
        if data is None:
            data = {}
        elif not isinstance(data, Tree):
            # If data is not a Tree try to cast to dict and iteratively recurse
            # it to convert each node to a tree
            data = dict(data)
            for k in data:
                if not isinstance(data[k], Tree) and isinstance(data[k], dict):
                    data[k] = Tree(data[k])
        # Add processed data to the tree
        super(Tree, self).__init__(Tree, data)
        if attr:
            self.update(attr)

    def __iter__(self, root=[]):
        it = collections.deque()
        for k_child, v_child in self.items():
            base = copy.copy(root)
            base.append(k_child)
            if isinstance(v_child, Tree):
                it.extend(v_child.__iter__(base))
            else:
                it.append((tuple(base), v_child))
        return iter(it)

    def __setitem__(self, k, v):
        if not isinstance(v, Tree) and isinstance(v, dict):
            v = Tree(v)
        super(Tree, self).__setitem__(k, v)

    def __reduce__(self):
        # This code is needed to fix an issue occurring while pickling.
        # Further info here:
        # http://stackoverflow.com/questions/3855428/how-to-pickle-and-unpickle-instances-of-a-class-that-inherits-from-defaultdict
        t = collections.defaultdict.__reduce__(self)
        return (t[0], ()) + t[2:]

    def __str__(self, dictonly=False):
        """Return a string representation of the tree

        Parameters
        ----------
        dictonly : bool, optional
            If True, just return a representation of a corresponding dictionary

        Returns
        -------
        tree : str
            A string representation of the tree
        """
        return "Tree({})".format(self.dict())

    @property
    def empty(self):
        """Return True if the tree is empty, False otherwise"""
        return len(self) == 0

    def update(self, e):
        """Update tree from e, similarly to dict.update

        Parameters
        ----------
        e : Tree
            The tree to update from
        """
        if not isinstance(e, Tree):
            e = Tree(e)
        super(Tree, self).update(e)

    def paths(self):
        """Return a dictionary mapping all paths to final (non-tree) values
        and the values.

        Returns
        -------
        paths : dict
            Path-value mapping
        """
        return dict(iter(self))

    def getval(self, path):
        """Get the value at a specific path, None if not there

        Parameters
        ----------
        path : iterable
            Path to the desired value

        Returns
        -------
        val : any type
            The value at the given path
        """
        tree = self
        for i in path:
            if isinstance(tree, Tree) and i in tree:
                tree = tree[i]
            else:
                return None
        return None if isinstance(tree, Tree) and tree.empty else tree

    def setval(self, path, val):
        """Set a value at a specific path

        Parameters
        ----------
        path : iterable
            Path to the value
        val : any type
            The value to set at the given path
        """
        tree = self
        for i in path[:-1]:
            if not isinstance(tree[i], Tree):
                tree[i] = Tree()
            tree = tree[i]
        tree[path[-1]] = val

    def dict(self, str_keys=False):
        """Convert the tree in nested dictionaries

        Parameters
        ----------
        str_key : bool, optional
            Convert keys to string. This is useful for example to dump a dict
            into a JSON object that requires keys to be strings

        Returns
        -------
        d : dict
            A nested dict representation of the tree
        """
        d = {}
        for k, v in self.items():
            k = str(k) if str_keys else k
            v = v.dict() if isinstance(v, Tree) else v
            d[k] = v
        return d

    def match(self, condition):
        """Check if the tree matches a given condition.

        The condition is another tree. This method iterates to all the values
        of the condition and verify that all values of the condition tree are
        present in this tree and have the same value.

        Note that the operation is not symmetric i.e.
        self.match(condition) != condition.match(self). In fact, this method
        return True if this tree has values not present in the condition tree
        while it would return False if the condition has values not present
        in this tree.

        Parameters
        ----------
        condition : Tree
            The condition to check

        Returns
        -------
        match : bool
            True if the tree matches the condition, False otherwise.
        """
        condition = Tree(condition)
        return all(self.getval(path) == val for path, val in condition.paths().items())

def parse_ashiip(path):
    """
    Parse a topology from an output file generated by the aShiip topology
    generator

    Parameters
    ----------
    path : str
        The path to the aShiip output file

    Returns
    -------
    topology : Topology
    """
    topology = fnss.Topology(type='ashiip')

    for line in open(path, "r").readlines():
        # There is no documented aShiip format but we assume that if the line
        # does not start with a number it is not part of the topology
        if line[0].isdigit():
            node_ids = re.findall("\d+", line)
            if len(node_ids) < 2:
                raise ValueError('Invalid input file. Parsing failed while ' \
                                 'trying to parse a line')
            node = int(node_ids[0])
            # level = int(node_ids[1])
            topology.add_node(node)
            for i in range(1, len(node_ids)):
                topology.add_edge(node, int(node_ids[i]))
    paths = dict(nx.all_pairs_dijkstra_path(topology))
    for u in paths:
        for v in paths[u]:
            paths[u][v] =len(paths[u][v])
    return paths

def cdf(data):
    """Return the empirical CDF of a set of 1D data

    Parameters
    ----------
    data : array-like
        Array of data

    Returns
    -------
    x : array
        All occurrences of data sorted
    cdf : array
        The CDF of data.
        More specifically cdf[i] is the probability that x < x[i]
    """
    if len(data) < 1:
        raise TypeError("data must have at least one element")
    freq_dict = collections.Counter(data)
    sorted_unique_data = np.sort(list(freq_dict.keys()))
    freqs = np.zeros(len(sorted_unique_data))
    for i in range(len(freqs)):
        freqs[i] = freq_dict[sorted_unique_data[i]]
#    freqs = np.array([freq_dict[sorted_unique_data[i]]
#                       for i in range(len(sorted_unique_data))])
    cdf = np.array(np.cumsum(freqs))
    norm = cdf[-1]
    cdf = cdf / norm  # normalize
    cdf[-1] = 1.0  # Prevent rounding errors
    return sorted_unique_data, cdf

def confidence_interval(data, beg, en):
    d = list(sorted(data))
    n = len(d)
    return d[int(beg*n)], d[int(en*n)]

if __name__ == '__main__':
    A = [1,2,5,3,4,6,7]
    print cdf(A)