"""
Script to run fitting all the logistic regressions
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),".."))

from data_api import *
from fit_lr import fit_lrs
from misc.config import c

n_jobs = 15

def exps():
    for id1, p in enumerate(proteins):
        for id2, f in enumerate(fingerprints):
            yield id1, id2


exps_list = list(exps())


def run(e):
    if proteins[e[0]] in c["EXCLUDED_PROTEINS"]:
	return
    print("Run "+str(e))

    config = {"protein":e[0], "fingerprint":e[1], "use_embedding":1, "max_hashes":1000, "K":30,
	      "experiment_name":"lr_prot_K_30_{0}_fin_{1}".format(*e)}

    if not os.path.exists(os.path.join(c["BASE_DIR"], config["experiment_name"]+".experiment")):
    	fit_lrs(config)


print(exps_list)


from multiprocessing import Pool
p = Pool(n_jobs)


p.map(run, exps_list)

