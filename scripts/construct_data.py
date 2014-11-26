"""
Script constructing embedded data
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),".."))

from data_api import fingerprints, proteins, prepare_experiment_data_embedded
from joblib import Parallel, delayed


n_jobs = 4


def exps():
    for id1, p in enumerate(proteins):
        for id2, f in enumerate(fingerprints):
            yield id1, id2


def run(e):
    print("Run "+str(e))
    D, config = prepare_experiment_data_embedded(protein=0, fingerprint=4, K=15, n_folds=10, seed=0, max_hashes=300, force_reload=True)



Parallel(n_jobs=n_jobs, backend="threading")(delayed(run)(e) for e in exps())

