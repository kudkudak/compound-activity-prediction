"""
Script constructing embedded data
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),".."))

from data_api import fingerprints, proteins, prepare_experiment_data_embedded, compute_jaccard_kernel


n_jobs = 12


def exps():
    for id1, p in enumerate(proteins):
        for id2, f in enumerate(fingerprints):
            yield id1, id2


def run(e):
    print("Run "+str(e))
    K = compute_jaccard_kernel(seed = 0, protein=e[0], fingerprint=e[1])


exps_list = list(exps())


from multiprocessing import Pool
p = Pool(n_jobs)


p.map(run, exps_list)

