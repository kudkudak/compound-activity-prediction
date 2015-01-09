# TODO: refactor to a class :)

from optparse import OptionParser
import cPickle
from config import c
import logging
import os


def get_exp_logger(config, to_file=False):
    name = config["experiment_name"]
    logger = logging.Logger(name=name, level=logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s \t" + name + "  %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if to_file:
        logger.propagate = False
        ch_file = logging.FileHandler(os.path.join(c["BASE_DIR"], name + ".log"))
        ch_file.setLevel(level=logging.INFO)
        ch_file.setFormatter(formatter)
        logger.addHandler(ch_file)
    return logger


def get_exp_options(config):
    config["experiment_name"] = "my_experiment"

    parser = OptionParser()
    parser.add_option("-e", "--e_name", dest="experiment_name", default="my_experiment")

    for cn, cv in config.iteritems():
        if type(cv) == type(1.0) or type(cv) == type(1) or type(cv) == type(""):
            parser.add_option("", "--" + cn, dest=cn, default=cv, type=type(cv))

    (options, args) = parser.parse_args()

    return {cn: getattr(options, cn) for cn in config.iterkeys()}


def print_exp_name(config):
    print "Experiment " + config["experiment_name"]


def print_exp_header(config):
    print "Experiment " + config["experiment_name"]
    print "======================="
    for cn, cv in config.iteritems():
        print "\t " + cn + " = " + str(cv)
    print "\n"


def generate_configs(c, grid):
    counts = [len(c[key]) for key in grid]
    id = [0] * len(grid)

    while id[0] != counts[0]:
        new_c = dict(c)
        for i, g in enumerate(grid):
            new_c[g] = c[g][id[i]]

        new_c["experiment_name"] += "_".join([grid[i] + "=" + str(new_c[grid[i]]) for i in range(len(grid))])
        yield new_c

        # Iterate index
        id[-1] += 1
        for i in range(len(grid) - 1, 0, -1):
            if id[i] == counts[i]:
                id[i - 1] += 1
                id[i] = 0
            else:
                break


def get_exp_fname(E):
    return E["config"]["experiment_name"] + ".experiment"


def exp_done(E):
    return os.path.exist(get_exp_fname(E))


def save_exp(E):
    cPickle.dump(E, open(os.path.join(c["BASE_DIR"], get_exp_fname(E)), "w"))
