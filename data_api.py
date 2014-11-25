#TODO: redo experiments with shuffle=true

from misc.utils import *
import sklearn
import glob
from sklearn.datasets import load_svmlight_file
from misc.lsh import *
import scipy
import itertools
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import jaccard_similarity_score




experiment_data = glob.glob(os.path.join(c["DATA_DIR"], "*.libsvm"))
fingerprints = ["EstateFP", "ExtFP", "KlekFP", "KlekFPCount", "MACCSFP", "PubchemFP", "SubFP", "SubFPCount"]
#proteins = list(set(["_".join(os.path.basename(i).split("_")[0:-1]) for i in experiment_data]))
proteins = ['5ht7','5ht6','SERT','5ht2c','5ht2a','hiv_integrase','h1','hERG','cathepsin','hiv_protease','M1','d2']
# Small test for loaded data
prot_counts = [len(glob.glob(os.path.join(c["DATA_DIR"], "*"+w+".libsvm"))) for w in fingerprints]
assert(all([i==12 for i in prot_counts]))



def construct_folds(protein=0, fingerprint=4, n_folds=10, seed=0):
    """
    Returns indexes of folds. Separated to make sure it is easily reproducible
    """
    np.random.seed(seed)
    X, Y = load_svmlight_file(os.path.join(c["DATA_DIR"], \
                                                       proteins[protein]+"_"+fingerprints[fingerprint]+".libsvm"))


    skf = sklearn.cross_validation.StratifiedKFold(Y, n_folds=n_folds, random_state = seed)
    folds = []
    for tr_id, ts_id in skf:
        folds.append({"train_id":tr_id, "test_id":ts_id})

    return folds


def set_representation_by_buckets(X):
    """
    Helper function transforms X into bucketed representation with features [x[j]>=bucket[j,i] for i,j in .. ]
    """

    # Investigate columns ranges
    ranges_max = X.max(axis=0).toarray().flatten()
    ranges_min = X.min(axis=0).toarray().flatten()
    spreads = [mx - mn for mx, mn in itertools.izip(ranges_max, ranges_min)]

    bucket_count = min(max(spreads), 5)

    # Iterate each column and creae evenly spread buckets
    buckets = []
    for col in X.T:
        col = np.array(col.toarray().reshape(-1))
        col.sort()
        col = col[np.where(col>0)[0][0]:]
        col_buckets = [a[0] for a in np.array_split(col, min(bucket_count, len(col)))]
        if len(col_buckets) < bucket_count:
            col_buckets += [col_buckets[-1]]*-(len(col_buckets)-bucket_count)
        buckets.append(col_buckets)

    # Create new matrix row by row
    feature_dict = {}

    X_tr = np.zeros(shape=(X.shape[0],bucket_count*X.shape[1]), dtype="int32")
    for i in range(X.shape[0]):
        row = []
        for col_idx, col in enumerate(X[i,:].toarray().reshape(-1)):
            # Faster than iterating X[i,:].T, iterating scipy sparse is slow
            for b,x in enumerate(buckets[col_idx]):
                assert(x <= ranges_max[col_idx])
                if(col >= x):
                    f = str(col_idx)+">= bucket_"+str(b)
                    if f not in feature_dict:
                        feature_dict[f] = len(feature_dict)
                    row.append(feature_dict[f])

        X_tr[i, row] = 1

    return scipy.sparse.csr_matrix(X_tr)



@cached_FS()
def get_mean_std_jaccard(protein, fingerprint):
    """
    Retrieves mean and std of jaccard distance within given protein-fingerprint sample

    Needed to estimate sensible threshold range (mean +- 1.5 std heuristic)
    """
    D, conf = prepare_experiment_data(n_folds=10, protein=protein, fingerprint=fingerprint)
    X_b = set_representation_by_buckets(D["X"])
    # Get similarity mean and deviation to get reasonable approximation of threshold
    h = []
    for i in range(1000):
        a,b = np.random.randint(low=0, high=X_b.shape[0], size=(2,))
        h.append(jaccard_similarity_score(X_b[a].toarray(), X_b[b].toarray()))
    m, std = np.array(h).mean(), np.array(h).std()
    return m,std


def construct_LSH_index(protein, fingerprint, n_folds=10, seed=0, threshold=0.56, max_hashes=200, set_representation_fnc = set_representation_by_buckets):
    """
    Constructs LSH index
    """

    X_bucketed = set_representation_fnc(prepare_experiment_data(protein=protein, fingerprint=fingerprint, n_folds=n_folds, seed=seed)[0]["X"])

    # Prepare objects
    C = Cluster(threshold=threshold, max_hashes=max_hashes)
    class WrapHashRow(object):
        def __init__(self, b):
            self.b = b
            self.hb = hash(str(self.b[0:min(20, len(self.b))]))
        def __hash__(self):
            return self.hb
        def __iter__(self):
            return iter(self.b)
    _, b = X_bucketed.nonzero()
    indptr = X_bucketed.indptr

    # Construct index
    for ex in range(X_bucketed.shape[0]):
        C.add(WrapHashRow(b[indptr[ex]:indptr[ex+1]]), label=ex)

    return C


@timed
@cached_FS()
def find_threshold(protein, fingerprint, K=15, n_folds=10):
    # Adapting threshold - this way rather than get_mean_std_jaccard as we are not assuming here gauss then
    f = fingerprint
    p = protein

    target = 2*K

    D, conf = prepare_experiment_data(n_folds=n_folds, protein=p, fingerprint=f)
    X_b = set_representation_by_buckets(D["X"])

    mean, std = get_mean_std_jaccard(protein=p, fingerprint=f)

    best_t = -1
    best_err = float("inf")

    thresholds = np.linspace(mean-1.5*std, mean+1.5*std, 50)
    for t in thresholds:
        print "Learning LSH index for ",t
        C = construct_LSH_index(protein=p, fingerprint=f, threshold=t, max_hashes=200, \
                                set_representation_fnc=set_representation_by_buckets)
        res = []
        # Sample 100 points
        for i in range(100):
            j = np.random.randint(low=0, high=X_b.shape[0])
            candidates = C.match(WrapHashRow(X_b[j].nonzero()[1]))
            res.append(len(candidates))
            #h = [jaccard_similarity_score(X_b[c].toarray(), X_b[j].toarray()) for c in candidates if c != j]

        err = float("inf")
        if len(candidates) > K:
            err = abs(len(candidates) - err)
            if err < best_err:
                best_t = t
                best_err = err

        print(np.array(res).mean())

    return best_t, best_err

def jaccard_statistics(X):
    X = set_representation_by_buckets(X)

    return X






@cached_FS()
def prepare_experiment_data(protein=0, fingerprint=4, n_folds=10, seed=0):
    np.random.seed(seed)
    X, Y = load_svmlight_file(os.path.join(c["DATA_DIR"], \
                                                       proteins[protein]+"_"+fingerprints[fingerprint]+".libsvm"))

    folds = construct_folds(protein=protein, fingerprint=fingerprint, n_folds=n_folds, seed=seed)
    D = {"folds": folds, "X":X, "Y":Y}, {"examples":X.shape[0]}

    return D
