from utils import *
import pandas as pd
from trainers import *
from data_api import *
from models import *
import cPickle
import glob
import matplotlib.pyplot as plt
import cPickle


#### Load params ####
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-e", "--experiment_name", dest="experiment_name", default="my_experiment")
(options, args) = parser.parse_args()

### Load experiment outputs ###
A = cPickle.load(open(options.experiment_name+".experiment", "r"))

### Print results ###
for monitor_name, monitor in A["monitors"].iteritems():
    print "\t" + monitor_name + "[-1]=" + str(monitor[-1])
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.set_xlabel('iteration')
    #ax.set_ylabel(monitor_name)
    plt.plot(range(len(monitor)), monitor)
    plt.show()

"""
for name, val in A["values"].iteritems():
    print name
    print "============"
    print val
    print "\n"
"""


