from misc.utils import *
import sklearn
import glob
from sklearn.datasets import load_svmlight_file

from sklearn.cross_validation import StratifiedKFold

experiment_data = glob.glob(os.path.join(c["DATA_DIR"], "*.libsvm"))

fingerprints = ["EstateFP", "ExtFP", "KlekFP", "KlekFPCount", "MACCSFP", "PubchemFP", "SubFP", "SubFPCount"]
#proteins = list(set(["_".join(os.path.basename(i).split("_")[0:-1]) for i in experiment_data]))

proteins = ['5ht7','5ht6','SERT','5ht2c','5ht2a','hiv_integrase','h1','hERG','cathepsin','hiv_protease','M1','d2']

# Small test for loaded data
prot_counts = [len(glob.glob(os.path.join(c["DATA_DIR"], "*"+w+".libsvm"))) for w in fingerprints]
assert(all([i==12 for i in prot_counts]))


import scipy
import itertools


def set_representation_by_buckets(X):

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
        col_buckets = [a[0] for a in np.array_split(col, min(5, len(col)))]
        if len(col_buckets) < bucket_count:
            col_buckets += [col_buckets[-1]]*-(len(col_buckets)-bucket_count)
        buckets.append(col_buckets)


    # Create new matrix row by row
    feature_dict = {}
    X_tr = np.zeros(shape=(X.shape[0],bucket_count*X.shape[1]), dtype="int32")
    for i in range(X.shape[0]):
        row = []
        for col_idx, col in enumerate(X[i,:].T):
            for b,x in enumerate(buckets[col_idx]):
                assert(x <= ranges_max[col_idx])
                if(col >= x):
                    f = str(col_idx)+">= bucket_"+str(b)
                    if f not in feature_dict:
                        feature_dict[f] = len(feature_dict)
                    row.append(feature_dict[f])

        X_tr[i, row] = 1

    return scipy.sparse.csr_matrix(X_tr)


def jaccard_statistics(X):
    X = set_representation_by_buckets(X)

    return X

@cached_FS()
def prepare_experiment_data(protein=0, fingerprint=4, n_folds=10, seed=0, transformation=None):

    np.random.seed(seed)
    X, Y = load_svmlight_file(os.path.join(c["DATA_DIR"], \
                                                       proteins[protein]+"_"+fingerprints[fingerprint]+".libsvm"))

    if transformation is not None:
        X = transformation(X)

    skf = sklearn.cross_validation.StratifiedKFold(Y, n_folds=n_folds)

    folds = []
    for tr_id, ts_id in skf:
        folds.append({"train_id":tr_id, "test_id":ts_id})

    return {"folds": folds, "X":X, "Y":Y}, {"examples":X.shape[0]}
