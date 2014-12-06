"""
Script to run fitting all the SVM-B RBF
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from misc.utils import *
from data_api import *
from fit_svms import fit_svms


n_jobs = 10

def exps():
    for id1, p in enumerate(proteins):
        for id2, f in enumerate(fingerprints):
            yield id1, id2


exps_list = list(exps())


def run(e):
    print("Run "+str(e))

    config = {"protein":e[0], "fingerprint":e[1], "kernel":"jaccard", "C_max":4,
              "experiment_name":"svm_jaccard_prot_{0}_fin_{1}".format(*e)}

    if not os.path.exists(os.path.join(c["BASE_DIR"], config["experiment_name"]+".experiment")):
        fit_svms(config)

from multiprocessing import Pool
p = Pool(n_jobs)



p.map(run, exps_list)

