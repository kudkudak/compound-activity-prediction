"""
Script constructing embedded data
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),".."))

from data_api import fingerprints, proteins, prepare_experiment_data_embedded


n_jobs = 10


def exps():
    for id1, p in enumerate(proteins):
        for id2, f in enumerate(fingerprints):
            yield id1, id2


def run(e):
    print("Run "+str(e))
    D, config = prepare_experiment_data_embedded(protein=e[0], fingerprint=e[1], K=30, n_folds=10, seed=0, max_hashes=1000)



exps_list = list(exps())


from multiprocessing import Pool
p = Pool(n_jobs)


p.map(run, exps_list)

