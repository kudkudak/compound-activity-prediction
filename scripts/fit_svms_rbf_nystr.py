"""
Script to run fitting all the SVM-B RBF
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from misc.utils import *
from data_api import fingerprints, proteins
from fit_svms import fit_svms


n_jobs = 15

def exps():
    for id1, p in enumerate(proteins):
        for id2, f in enumerate(fingerprints):
            yield id1, id2


exps_list = list(exps())


def run(e):
    print("Run "+str(e))

    config = {"protein":e[0], "fingerprint":e[1], "kernel":"rbf-nystroem", "C_max":4, "gamma_max":-1, "gamma_min":-10,
              "experiment_name":"svm_rbf_nystr_prot_{0}_fin_{1}".format(*e)}

    if not os.path.exists(os.path.join(c["BASE_DIR"], config["experiment_name"]+".experiment")):
        fit_svms(config)


print(exps_list)

from multiprocessing import Pool
p = Pool(n_jobs)


p.map(run, exps_list)

