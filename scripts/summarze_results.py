"""
Usage python summarize_results.py prefix model model_name
"""


import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from misc.utils import *
from data_api import *
from misc.experiment_utils import *
import fnmatch
import sys
import cPickle

prefix = sys.argv[1] #For instance svm_rbf_nystr
model = sys.argv[2]
out = sys.argv[3]
model_name = sys.argv[2] # Usually prefix

start_folder = ""
if len(sys.argv) > 4:
    start_folder = sys.argv[4]

assert(model in ["svm_jaccard","svm_rbf_nystr", "svm_rbf", "KNN", "KNN_LSH", "svm_linear"])

extract_configs = {
    "svm_jaccard": ["C"],
    "svm_rbf_nystr": ["C", "gamma"],
    "svm_rbf":["C", "gamma"],
    "KNN":["KNN_K"],
    "KNN_LSH": [],
    "svm_linear": ["C"]
}

results_table = ["mean_mcc", "mean_wac", "mean_acc"]
monitors_mean = ["train_time", "test_time", "n_support"]
# Get experiments wanted

experiments = []
for root, dirnames, filenames in os.walk(os.path.join(c["BASE_DIR"],start_folder)):
  for filename in fnmatch.filter(filenames, prefix+'*.experiment'):
     experiments.append(os.path.join(root, filename))

# Few necessary hacks :(
import re
if model == "KNN":
    experiments = [e for e in experiments if not re.match(".*LSH.*", e)]
if model == "svm_rbf":
    experiments = [e for e in experiments if not re.match(".*nystr.*", e)]

print experiments

# Create list of all experiments

def exps():
    for id1, p in enumerate(proteins):
        for id2, f in enumerate(fingerprints):
            yield id1, id2



# Load experiments and group by prot/fin
experiments_grouped = {}

for e_name in experiments:
    try:
        E = cPickle.load(open(e_name,"r"))
        logger = get_exp_logger(E["config"])

        ## Find best experiment ##
        best_e = E["experiments"][0]
        for e in E["experiments"]:
            if e["results"]["mean_wac"] > best_e["results"]["mean_wac"]:
                best_e = e

        print best_e["config"]

        experiments_grouped[(best_e["config"]["protein"], best_e["config"]["fingerprint"])] = best_e
    except:
        pass



# For each protein write out statistics
results = []
for prot, fin in exps():
    if (prot, fin) in experiments_grouped:
        row = [proteins[prot]+":"+fingerprints[fin]] + \
               [model + " " +" ".join([k+":"+str(experiments_grouped[(prot,fin)]["config"][k]) for k in extract_configs[model]])]
        for r in results_table:
            row.append(experiments_grouped[(prot,fin)]["results"].get(r, np.nan))
        for m in monitors_mean:
            row.append(np.array(experiments_grouped[(prot,fin)]["monitors"].get(m, [np.nan])).mean())
        results.append(row)
    else:
        results.append([proteins[prot]+":"+fingerprints[fin]] + [""] + [np.nan]*len(results_table) \
            + [np.nan]*len(monitors_mean))
    print results[-1]


frame = pd.DataFrame(results, columns = ["experiment"] + ["params"] + results_table + monitors_mean)

frame.to_csv(os.path.join(c["BASE_DIR"], out+".csv"))
