"""
Script constructing embedded data
"""

n_jobs = 4

import os
from data_api import fingerprints, proteins
from fit_svms import fit_svms
from joblib import Parallel, delayed

def exps():
    for id1, p in enumerate(proteins):
        for id2, f in enumerate(fingerprints):
            yield id1, id2


def run(e):
    print("Run "+str(e))

    config = {"protein":e[0], "fingerprint":e[1],\
              "experiment_name":"svm_rbf_prot_{0}_fin_{1}".format(*e)}

    if not os.path.exists(config["experiment_name"]+".experiment"):
        fit_svms(config)


Parallel(n_jobs=n_jobs, backend="threading")(delayed(run)(e) for e in exps())

