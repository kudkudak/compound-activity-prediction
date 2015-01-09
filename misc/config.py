'''
Basic config for whole project. Modify those variables to reflect changesG
'''
import os
import sys
import logging





import os
base_dir = "/home/moje/Projekty_big/tfml-melc"
name = "melc"




# Logger
def get_logger(name):
    logging.basicConfig(level = logging.INFO)
    logger = logging.getLogger(name)
    ch = logging.StreamHandler()
    formatter = logging.Formatter(name+': %(asctime)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False
    ch_file = logging.FileHandler(os.path.join(base_dir, name + ".log"))
    ch_file.setLevel(level = logging.INFO)
    ch_file.setFormatter(formatter)
    logger.addHandler(ch_file)
    return logger

logger = get_logger("tfml_melc")



# Configurations
c = {
    # Time in seconds after which fit_svms.py learning is interrupted
    "OPTIMIZED_MEASURE": "mean_wac",
    "EXCLUDED_PROTEINS": set(["hiv_protease", "d2"]),
    "MAX_TRAIN_TIME_SVM" : 100 ,
    "CACHE_DIR" : os.path.join(base_dir, "cache"),
    "DATA_DIR":os.path.join(base_dir, "data"),
    "BASE_DIR":base_dir,
    "CURRENT_EXPERIMENT_CONFIG":{"experiment_name":"my_favourite_experiment"}
}
