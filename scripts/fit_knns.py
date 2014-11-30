"""
Script to run fitting all the SVM-B RBF
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),".."))

from data_api import fingerprints, proteins
from fit_knn import fit_knns


n_jobs = 10

def exps():
    for id1, p in enumerate(proteins):
        for id2, f in enumerate(fingerprints):
            yield id1, id2


exps_list = list(exps())


def run(e):
    print("Run "+str(e))

    config = {"protein":e[0], "fingerprint":e[1], "use_embedding":0,
              "experiment_name":"KNN_{0}_fin_{1}".format(*e)}

    if not os.path.exists(config["experiment_name"]+".experiment"):
        fit_knns(config)


    config = {"protein":e[0], "fingerprint":e[1], "use_embedding":1, "max_hashes":1000, "K":20,
              "experiment_name":"KNN_LSH_{0}_fin_{1}".format(*e)}

    if not os.path.exists(config["experiment_name"]+".experiment"):
        fit_knns(config)


from multiprocessing import Pool
p = Pool(n_jobs)


p.map(run, exps_list)

