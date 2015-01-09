
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from misc.utils import *
from data_api import *
from misc.experiment_utils import *
import fnmatch
import sys
import cPickle

statistics = ["mean_acc", "mean_wac", "mean_acc"] +  ["train_time", "test_time", "n_support"]

import glob

results = glob.glob("*.csv")


