"""
Script to run fitting all the SVM-B RBF
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),".."))

from data_api import *
from fit_svms import fit_svms
from misc.config import c

n_jobs = 6

def exps():
    for id1, p in enumerate(proteins):
        for id2, f in enumerate(fingerprints):
            yield id1, id2


exps_list = list(exps())


def run(e):
    if proteins[e[0]] not in c["EXCLUDED_PROTEINS"]: 
	    print("Run "+str(e))

	    config = {"protein":e[0], "fingerprint":e[1], "use_embedding":1, "kernel":"linear", "max_hashes":1000, "K":30, "C_max":5,
		      "experiment_name":"svm_linear_fixed_K_30_{0}_fin_{1}".format(*e)}

	    if not os.path.exists(os.path.join(c["BASE_DIR"], config["experiment_name"]+".experiment")):
		fit_svms(config)


print(exps_list)


from multiprocessing import Pool
p = Pool(n_jobs)


p.map(run, exps_list)

