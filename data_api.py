from misc.utils import *
import sklearn
import glob
from sklearn.datasets import load_svmlight_file

experiment_data = glob.glob(os.path.join(c["DATA_DIR"], "*.libsvm"))

fingerprints = ["EstateFP", "ExtFP", "KlekFP", "KlekFPCount", "MACCSFP", "PubchemFP", "SubFP", "SubFPCount"]
proteins = list(set([os.path.basename(i).split("_")[0] for i in experiment_data]))


# Small test for loaded data
prot_counts = [len(glob.glob(os.path.join(c["DATA_DIR"], "*"+w+".libsvm"))) for w in fingerprints]
assert(all([i==12 for i in prot_counts]))





def prepare_experiment_data(protein=0, fingerprint=4, n_folds=10, seed=0):
    np.random.seed(seed)
    X, Y = load_svmlight_file(os.path.join(c["DATA_DIR"], \
                                                       proteins[protein]+"_"+fingerprints[fingerprint]+".libsvm"))
    skf = sklearn.cross_validation.StratifiedKFold(Y, n_folds=n_folds)

    folds = []
    for tr_id, ts_id in skf:
        folds.append({"train_id":tr_id, "test_id":ts_id})

    return {"folds": folds, "X":X, "Y":Y}, {}
