"""
Script to run fitting all the SVM-B RBF
"""
import sys
import os
from data_api import fingerprints, proteins
from fit_svms import fit_svms
from joblib import Parallel, delayed
sys.path.append(os.path.join(os.path.dirname(__file__),".."))


n_jobs = 4

def exps():
    for id1, p in enumerate(proteins):
        for id2, f in enumerate(fingerprints):
            yield id1, id2


exps_list = list(exps())


def run(e):
    print("Run "+str(e))

    config = {"protein":e[0], "fingerprint":e[1],\
              "experiment_name":"svm_rbf_prot_{0}_fin_{1}".format(*e)}

    if not os.path.exists(config["experiment_name"]+".experiment"):
        fit_svms(config)


print(exps_list)

from multiprocessing import Pool
p = Pool(n_jobs)


p.map(run, exps_list)

