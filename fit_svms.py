from misc.utils import *

from misc.experiment_utils import get_exp_options, print_exp_header, save_exp, get_exp_logger, generate_configs
from data_api import prepare_experiment_data

from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.svm import SVC

#### Load config and data ####
config = {"protein":0, "fingerprint":4,"n_folds":10}
config.update(get_exp_options(config))
D, config_from_data = prepare_experiment_data(n_folds=10, protein=config["protein"], fingerprint=config["fingerprint"])
config.update(config_from_data)
config["C"] = np.linspace(1, 1000, 10)
config["gamma"] = [1e-3, 1e-4, 1e-5]
logger = get_exp_logger(config)



def fit_svm(config):
    ### Prepare result holders ###
    values = {}
    results = {}
    monitors = {}
    E = {"config": config, "results": results, "monitors":monitors, "values":values}

    ### Print experiment header ###
    print_exp_header(config)

    ### Train ###
    monitors["acc_fold"] = []
    monitors["mcc_fold"] = []

    results["mean_acc"] = 0
    results["mean_mcc"] = 0

    X, Y = D["X"], D["Y"]
    values["mean_cls"] = Y.mean()
    for fold in D["folds"]:
        tr_id, ts_id = fold["train_id"], fold["test_id"]
        X_train, Y_train, X_test, Y_test = X[tr_id], Y[tr_id], X[ts_id], Y[ts_id]
        m = SVC(C=config["C"], gamma=config["gamma"])
        m.fit(X_train, Y_train)
        acc_fold, mcc_fold = accuracy_score(m.predict(X_test), Y_test), matthews_corrcoef(m.predict(X_test), Y_test)
        monitors["acc_fold"].append(acc_fold)
        monitors["mcc_fold"].append(mcc_fold)

    monitors["acc_fold"] = np.array(monitors["acc_fold"])
    monitors["mcc_fold"] = np.array(monitors["mcc_fold"])

    results["mean_acc"] = monitors["acc_fold"].mean()
    results["mean_mcc"] = monitors["mcc_fold"].mean()

    logger.info(results)

    return E


cv_configs = generate_configs(config, ["C", "gamma"])

for c in cv_configs:
    fit_svm(c)
