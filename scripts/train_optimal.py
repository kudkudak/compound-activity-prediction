"""
Trains all optimal models of prefix and stores them i prefix.pkl file
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from misc.utils import *
from data_api import *
from misc.experiment_utils import *
import fnmatch
import sys
from fit_svms import fit_svm
import cPickle
import glob
prefix = sys.argv[1] #For instance svm_rbf_nystr

@cached_FS()
def get_experiments(prefix):

    experiments = glob.glob(os.path.join(c["BASE_DIR"], prefix+"*.experiment"))

    experiments_grouped = {}
    for e_name in experiments:
        try:
            E = cPickle.load(open(e_name,"r"))
            logger = get_exp_logger(E["config"])

            ## Find best experiment ##
            best_e = E["experiments"][0]

            if proteins[best_e["config"]["protein"]] in c["EXCLUDED_PROTEINS"]:
                continue

            for e in E["experiments"]:
                if e["results"][c["OPTIMIZED_MEASURE"]] > best_e["results"][c["OPTIMIZED_MEASURE"]]:
                    best_e = e

            print best_e["config"]

            experiments_grouped[(best_e["config"]["protein"], best_e["config"]["fingerprint"])] = best_e
        except:
            pass

    return experiments_grouped

experiments_grouped = get_experiments(prefix=prefix)
learned = {}

def run(e):
    prot = e[0]
    fin = e[1]

    if proteins[prot] in c["EXCLUDED_PROTEINS"]:
        return

    if not (prot, fin) in experiments_grouped:
	return

    exp = experiments_grouped[(prot, fin)]

    print "Fetching necessary data"
    X, Y = get_raw_training_data(protein=prot, fingerprint=fin)

    config = exp["config"]
    if config["use_embedding"] == 1:
        D, _ = prepare_experiment_data_embedded(protein=prot, fingerprint=fin,K=config["K"],
                                                max_hashes=config["max_hashes"], seed=0,n_folds=10)
    else:
        D, _ = prepare_experiment_data(protein=prot, fingerprint=fin, seed=0, force_reload=True)
	
    assert(config["protein"] == e[0])
    assert(config["fingerprint"] == e[1])

    if config["kernel"] == "jaccard":
	K = compute_jaccard_kernel(protein=config["protein"], fingerprint=config["fingerprint"], seed=config["seed"])
    else:
	K = []
    config["store_clf"] = True
    print "Learning SVM"
    E = fit_svm(exp["config"], D, X, Y, K)
    path = os.path.join(c["BASE_DIR"], prefix+"_model_"+str(e[0])+"_"+str(e[1])+".optimal")

    if not os.path.exists(path):
    	cPickle.dump(E, open(path, "w"))



n_jobs=5
from multiprocessing import Pool
p = Pool(n_jobs)
p.map(run, list(exps()))

import cPickle


