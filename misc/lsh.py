"""
Code is based on Florian Leitner Text Mining course

http://nbviewer.ipython.org/github/fnl/asdm-tm-class/blob/master/Locality%20Sensitive%20Hashing.ipynb

Whole course can be found here:

https://github.com/fnl/asdm-tm-class

Thanks!
"""


from collections import defaultdict
import numpy as np
import math

def pick_bands(threshold, hashes):
    target = hashes * -1 * math.log(threshold)
    bands = 1
    while bands * math.log(bands) < target:
        bands += 1
    return bands

def pick_hashes_and_bands(threshold, max_hashes):
    bands = pick_bands(threshold, max_hashes)
    hashes = (max_hashes / bands) * bands
    return (hashes, bands)

def pick_hashes_and_bands2(threshold, max_hashes):
    best_err = 1e6
    best_b = -1
    target = 0.5
    for b in range(1, max_hashes):
        r = max_hashes/b
        p_at_tr = 1. - (1. - threshold**r)**b
        if abs(p_at_tr - target) < best_err:
            best_err = abs(p_at_tr - target)
            best_b = b
    return (max_hashes/best_b)*best_b, best_b


class WrapHashRow(object):
    def __init__(self, b):
        self.b = b
        self.hb = hash(str(self.b[0:min(20, len(self.b))]))

    def __hash__(self):
        return self.hb

    def __iter__(self):
        return iter(self.b)

# Jak to przyspieszyc?
class SparseMinHashSignature:
    """Hash signatures for sets/tuples using minhash."""

    def __init__(self, dim):
        """
        Define the dimension of the hash pool
        (number of hash functions).
        """
        self.dim = dim
        self.a = np.random.randint(low=3000, high=1e9, size=(dim,2))
        self.p = 805306457


    def sign(self, item):
        """Return the minhash signatures for the `item`."""
        sig = [ float("inf") ] * self.dim
        for i in xrange(self.a.shape[0]):
            # minhashing; requires item is iterable:
            sig[i] = ((self.a[i,0]*item + self.a[i,1])%self.p).min()

        return sig

class LSH:
    """
    Locality sensitive hashing.

    Uses a banding approach to hash
    similar signatures to the same buckets.
    """

    def __init__(self, size, bands, threshold):
        """
        LSH approximating a given similarity `threshold`
        with a given hash signature `size`.
        """
        self.size = size
        self.threshold = threshold
        self.bands = bands
        self.bandwidth = self.size/self.bands


    def hash(self, sig):
        """Generate hash values for this signature."""
        for band in zip(*(iter(sig),) * self.bandwidth):
            yield hash("salt" + str(band) + "tlas")

    @property
    def exact_threshold(self):
        """The exact threshold defined by the chosen bandwith."""
        r = self.bandwidth
        b = self.size / r
        return (1. / b) ** (1. / r)

    def get_n_bands(self):
        """The number of bands."""
        return int(self.size / self.bandwidth)

class UnionFind:
    """
    Union-find data structure.

    Each unionFind instance X maintains a family of disjoint sets of
    hashable objects, supporting the following two methods:

    - X[item] returns a name for the set containing the given item.
    Each set is named by an arbitrarily-chosen one of its members; as
    long as the set remains unchanged it will keep the same name. If
    the item is not yet part of a set in X, a new singleton set is
    created for it.

    - X.union(item1, item2, ...) merges the sets containing each item
    into a single larger set. If any item is not yet part of a set
    in X, it is added to X as one of the members of the merged set.

    Source: http://www.ics.uci.edu/~eppstein/PADS/UnionFind.py

    Union-find data structure. Based on Josiah Carlson's code,
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/215912
    with significant additional changes by D. Eppstein.
    """

    def __init__(self):
        """Create a new empty union-find structure."""
        self.weights = {}
        self.parents = {}

    def __getitem__(self, object):
        """Find and return the name of the set containing the object."""
        # check for previously unknown object
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]

        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root

        return root

    def __iter__(self):
        """Iterate through all items ever found or unioned by this structure."""
        return iter(self.parents)

    def union(self, *objects):
        """Find the sets containing the objects and merge them all."""
        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r],r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest

    def sets(self):
        """Return a list of each disjoint set"""
        ret = defaultdict(list)
        for k, _ in self.parents.iteritems():
            ret[self[k]].append(k)
        return ret.values()


class Cluster:
    """
    Cluster items with a Jaccard similarity above
    some `threshold` with a high probability.

    Based on Rajaraman, "Mining of Massive Datasets":

    1. Generate items hash signatures
    2. Use LSH to map similar signatures to same buckets
    3. Use UnionFind to merge buckets containing same values
    """

    def __init__(self, threshold=0.5, max_hashes=50,  MinHasher=SparseMinHashSignature):
        """
        The `size` parameter controls the number of hash
        functions ("signature size") to create.
        """
        self.hashes, self.bands = pick_hashes_and_bands2(threshold, max_hashes)
        print "Picked "+str(self.hashes)+ " and bands=" + str(self.bands)

        self.unions = UnionFind()
        self.signer = MinHasher(self.hashes)
        self.hasher = LSH(self.hashes, self.bands, threshold)
        self.hashmaps = [
            defaultdict(list) for _ in range(self.hasher.get_n_bands())
        ]

        self.signature_cache = {}

    def add(self, item, label):
        """
        Add an `item` to the cluster.

        Optionally, use a `label` to reference this `item`.
        Otherwise, the `item` itself is used as the label.
        """
        # Add to unionfind structure
        self.unions[label]

        # Get item signature
        if label not in self.signature_cache:
            sig = self.signer.sign(item)
            self.signature_cache[label] = sig
        else:
            sig = self.signature_cache[label]

        # Unite labels with the same LSH keys in the same band
        for band_idx, hashval in enumerate(self.hasher.hash(sig)):
            self.hashmaps[band_idx][hashval].append(label)
            self.unions.union(label, self.hashmaps[band_idx][hashval][0])


    def groups(self):
        """
        Get the clustering result.

        Returns sets of labels.
        """
        return self.unions.sets()

    def match(self, item, label):
        """
        Get a set of matching labels for `item`.

        Returns a (possibly empty) set of labels.
        """

        # Get signature (we cache signatue when given label)
        if label is None:
            sig = self.signer.sign(item)
        else:
            if label in self.signature_cache:
                sig = self.signature_cache[label]
            else:
                sig = self.signer.sign(item)
                self.signature_cache[label] = sig


        matches = set()

        for band_idx, hashval in enumerate(self.hasher.hash(sig)):
            if hashval in self.hashmaps[band_idx]:
                matches.update(self.hashmaps[band_idx][hashval])

        return matches