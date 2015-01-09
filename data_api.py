#TODO: redo experiments with shuffle=true
#TODO: make unit tests checking how many I have missed as %
#test

# Cel - dobrac tak target, aby praktycznie nie wystepowalo mniej niz K kandydatow

from misc.utils import *
import sklearn
import glob
from sklearn.datasets import load_svmlight_file
from misc.lsh import *
import scipy
import itertools
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import jaccard_similarity_score, confusion_matrix
import itertools



experiment_data = glob.glob(os.path.join(c["DATA_DIR"], "*.libsvm"))
fingerprints = ["EstateFP", "ExtFP", "KlekFP", "KlekFPCount", "MACCSFP", "PubchemFP", "SubFP", "SubFPCount"]
#proteins = list(set(["_".join(os.path.basename(i).split("_")[0:-1]) for i in experiment_data]))
proteins = ['5ht7','5ht6','SERT','5ht2c','5ht2a','hiv_integrase','h1','hERG','cathepsin','hiv_protease','M1','d2']
# Small test for loaded data
prot_counts = [len(glob.glob(os.path.join(c["DATA_DIR"], "*"+w+".libsvm"))) for w in fingerprints]
assert(all([i==12 for i in prot_counts]))

def wac_score(Y_true, Y_pred):
    cm = confusion_matrix(Y_true, Y_pred)
    tp, fn, fp, tn = cm[1,1], cm[1,0], cm[0,1], cm[0,0]
    return 0.5*tp/float(tp+fn) + 0.5*tn/float(tn+fp)

def jaccard_similarity_score_fast(r1, r2):
    dt = float(r1.dot(r2.T).sum())
    return dt / (r1.sum() + r2.sum() - dt )

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




def construct_LSH_index(protein=0, fingerprint=4, threshold=0.56, max_hashes=200):
    """
    Constructs LSH index
    """
    # Prepare objects

    X, Y = load_svmlight_file(os.path.join(c["DATA_DIR"], proteins[protein]+"_"+fingerprints[fingerprint]+".libsvm"))
    X = set_representation_by_buckets(X)

    C = Cluster(threshold=threshold, max_hashes=max_hashes)

    # Construct index
    for ex in range(X.shape[0]):
        C.add(X[ex].nonzero()[1], label=ex) #b[indptr[ex]:indptr[ex+1]]

    return C


STANDARD = 0
FIX_SCALING = 1

@timed
@cached_FS(skip_args=["collect_statistics"])
def prepare_experiment_data_embedded(protein=0, fingerprint=4, K=15, \
                                     n_folds=10, max_hashes=300, seed=0, limit=None,
                                     representation_version = STANDARD,
                                     collect_statistics=False, experimental=False, calculate_folds=range(10)):
    """
    Prepares experiment data embedded using jaccard similarity
    """

    # Load and transform data to buckets
    X, Y = load_svmlight_file(os.path.join(c["DATA_DIR"], proteins[protein]+"_"+fingerprints[fingerprint]+".libsvm"))
    X = set_representation_by_buckets(X)


    # Construct sequential lsh indexes labeled by row id. We need those because distance distribution is
    # not well gaussian
    # lsh_thresholds= [0.3,0.4,0.45, 0.5, 0.55, 0.6, 0.65,0.7,0.75, 0.8,0.9]

    print "Constructing lsh_indexes"
    #lsh_indexes = [construct_LSH_index(protein=protein, fingerprint=fingerprint, threshold=t, max_hashes=max_hashes\
    #     ) for t in lsh_thresholds]
    #
    # if experimental:
    #     # Try this
    lsh_indexes = [construct_LSH_index(protein=protein, fingerprint=fingerprint, threshold=i, max_hashes=max_hashes) \
            for i in np.linspace(0.2, 0.9, 14)]
    #
    # else:


    #lsh_thresholds= [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,0.7, 0.8, 0.9]


    print "Constructing lsh_indexes"


    #lsh_indexes = [construct_LSH_index(protein=protein, fingerprint=fingerprint, threshold=t, max_hashes=max_hashes\
    #     ) for t in lsh_thresholds]

    # Prepare folds ids
    print "Constructing fold indexes"
    folds_idx = construct_folds(protein=protein, fingerprint=fingerprint, n_folds=n_folds, seed=seed)
    folds = []



    statistics_len_split = []
    statistics_len_candidates = []
    whole_scans = [0]
    picked_threshold = [0]

    # Collected if experimental is ON
    dists_pos_pos = []
    dists_pos_neg = []
    dists_neg_pos = []
    dists_neg_neg = []

    # Constructing folds
    for fold_id in calculate_folds:
        fold = folds_idx[fold_id]
        tr_id, ts_id = fold["train_id"], fold["test_id"]

        Y_train, Y_test = Y[tr_id],  Y[ts_id]
        tr_id_set, ts_id_set = set(tr_id), set(ts_id) #sets for fast filtering

        X_train_lsh, X_test_lsh = [], [] # We will construct them row by row

        @timed
        def construct_embedding(source, target):
            # Construct train data

            itr = source

            if limit != None:
                itr = itertools.islice(source, limit)

            for row_idx in itr:
                # Query LSHs
                candidates = [list(index.match(X[row_idx].nonzero()[1], label=row_idx)) for index in lsh_indexes]

                # Pick closest
                best = []
                best_err = float('inf')
                best_id = -1
                for id, c in enumerate(reversed(candidates)):
                    if len(c) > 2*K and abs(len(c) - 2*K) < best_err:
                        best_err = abs(len(c) - 2*K)
                        best = list(c)
                        best_id=id

                    if len(c) > 2.5*K:
                        # this is an imporant heuristic - if it is a reasonably big set accept bigger threshold
                        best = list(c)
                        best_id=id
                        break

                picked_threshold.append(best_id)

                # Basic filtering (we can look only at training examples)
                candidates = [c for c in best if c != row_idx and c in tr_id_set]

                statistics_len_candidates.append(len(candidates))

                candidates_sims = []

                if len(best): # <=> we found a good LSH threshold
                    # Caching result
                    candidates_sims = np.array([jaccard_similarity_score_fast(X[row_idx], X[idx]) \
                                       for idx in candidates])
                else:
                    candidates_sims = np.array([jaccard_similarity_score_fast(X[row_idx], X[idx]) \
                                       for idx in tr_id])
                    candidates = tr_id
                    whole_scans[0] += 1

                # Sort and get K closests in relative indexes (for fast query, optimization)
                candidates_relative = sorted(range(len(candidates))\
                                             , key=lambda idx: -candidates_sims[idx] )[0:K] # decreasing by default so reverse

                # Get dists
                candidates_pos_dists = np.array([candidates_sims[idx] for idx in candidates_relative if Y[candidates[idx]]==1])
                candidates_neg_dists = np.array([candidates_sims[idx] for idx in candidates_relative if Y[candidates[idx]]==-1])

                if collect_statistics:
                    if Y[row_idx] == 1:
                        dists_neg_pos.append(candidates_neg_dists)
                        dists_pos_pos.append(candidates_pos_dists)

                    if Y[row_idx] == -1:
                        dists_neg_neg.append(candidates_neg_dists)
                        dists_pos_neg.append(candidates_pos_dists)

                if representation_version == FIX_SCALING:
                    target.append([len(candidates_pos_dists)/(0.1 + float(len(candidates_relative))), \
                                   len(candidates_neg_dists)/(0.1 + float(len(candidates_relative))), \
                                   candidates_pos_dists.mean() if len(candidates_pos_dists) else 0.0, \
                                   candidates_pos_dists.min() if len(candidates_pos_dists) else 0.0, \
                                   candidates_pos_dists.max() if len(candidates_pos_dists) else 0.0,\
                                   candidates_neg_dists.mean() if len(candidates_neg_dists) else 0.0, \
                                   candidates_neg_dists.max() if len(candidates_neg_dists) else 0.0, \
                                   candidates_neg_dists.min() if len(candidates_neg_dists) else 0.0] )
                elif representation_version == STANDARD:
                     target.append([len(candidates_pos_dists), \
                                   len(candidates_neg_dists), \
                                   candidates_pos_dists.mean() if len(candidates_pos_dists) else 0.0, \
                                   candidates_pos_dists.min() if len(candidates_pos_dists) else 0.0, \
                                   candidates_pos_dists.max() if len(candidates_pos_dists) else 0.0,\
                                   candidates_neg_dists.mean() if len(candidates_neg_dists) else 0.0, \
                                   candidates_neg_dists.max() if len(candidates_neg_dists) else 0.0, \
                                   candidates_neg_dists.min() if len(candidates_neg_dists) else 0.0] )

        construct_embedding(list(tr_id), X_train_lsh)
        print "Calculating ",protein, fingerprint
        construct_embedding(list(ts_id), X_test_lsh)

        folds.append({"X_train": np.array(X_train_lsh), "X_test":np.array(X_test_lsh), "Y_train":Y_train, "Y_test":Y_test})

    return {"folds":folds, "folds_idx":folds_idx, \
            "len_candidates":statistics_len_candidates, "len_split": statistics_len_split,\
            "whole_scans": whole_scans[0], "picked_threshold":picked_threshold,
            }, {"examples": X.shape[0]}




@cached_FS()
def prepare_experiment_data(protein=0, fingerprint=4, n_folds=10, seed=0):
    """
    Prepares experiment data for RBF case
    """
    np.random.seed(seed)
    X, Y = load_svmlight_file(os.path.join(c["DATA_DIR"], \
                                                       proteins[protein]+"_"+fingerprints[fingerprint]+".libsvm"))

    folds = construct_folds(protein=protein, fingerprint=fingerprint, n_folds=n_folds, seed=seed)
    D = {"folds": folds, "X":X, "Y":Y}, {"examples":X.shape[0]}

    return D

@cached_FS()
def get_raw_training_data(protein, fingerprint, n_folds=10, seed=0):
    """
    Prepares experiment data for RBF case
    """
    np.random.seed(seed)
    X, Y = load_svmlight_file(os.path.join(c["DATA_DIR"], \
                                                       proteins[protein]+"_"+fingerprints[fingerprint]+".libsvm"))
    return X, Y


@timed
@cached_FS()
def compute_jaccard_kernel(protein, fingerprint, seed):
    D, _ = prepare_experiment_data(protein=protein, fingerprint=fingerprint, n_folds=10, seed=seed)
    X = D["X"]
    print X.shape
    X = set_representation_by_buckets(X)
    print X.shape
    K = np.zeros(shape=(X.shape[0], X.shape[0]))
    print K.shape
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            K[i,j] = jaccard_similarity_score_fast(X[i], X[j])
    return K


# Create list of all experiments

def exps():
    for id1, p in enumerate(proteins):
        for id2, f in enumerate(fingerprints):
            yield id1, id2

"""
Notatki do publikacji

* Problem z gaussowscia rozkladow, - co jak sa 2 gaussy. Sekwencyjne LSH
* Dobor thresholda, optymalizacja
* Statystyki otocznia optymalizowane tak zeby klasy byly rozne
* Najwazniejszy jest najblizszy - wazne zeby go dobrze znalezc: uczymy sie na zbiorze testujacym! KNN mocno spada z K.
Ale nie zawsze! Sa takie ze sie liczy 3 (fingerprint = 0/0 np)
* Stala liczba przegladnaych kandydatow na zbiorach od 1k do 4k - teoretycznie tak powinno zostac
* RBF noestrem - musi miec udze C

TODO:
ok; mozna puscic melc tam i zobaczyc co sie stanie
dwa -> zobaczyc czy jednak nie jest za male K, dac np 20
trzy -> dodac to PCA (do 95% wariancji np)
no to mozesz dac 10 i 20 zeby zobaczyc czy nie za duzo/malo
i te 3 rzeczy powinny dac intuicje co dalej
cztery -> sprawdzic knn

"""
