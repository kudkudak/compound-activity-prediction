from misc.utils import *
from misc.experiment_utils import *
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),".."))

## Load experiments ##
import sys
import cPickle
E = cPickle.load(open(sys.argv[1],"r"))
logger = get_exp_logger(E["config"])

## Print best mcc ##
best_e = E["experiments"][0]
for e in E["experiments"]:
    if e["results"]["mean_mcc"] > best_e["results"]["mean_mcc"]:
        best_e = e
logger.info(best_e)
logger.info(best_e["results"]["mean_mcc"])

