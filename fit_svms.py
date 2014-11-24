from misc.utils import *

from misc.experiment_utils import get_exp_options, print_exp_header, \
    save_exp, get_exp_logger, generate_configs, print_exp_name
from data_api import prepare_experiment_data

from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler

def fit_svms(config_in = None):
    #### Load config and data ####
    config = {"protein":0, "fingerprint":4,"n_folds":10, "grid_w":10, "kernel":"rbf"}
    if config_in is None:
        config.update(get_exp_options(config))
    else:
        config.update(config_in)

    D, config_from_data = prepare_experiment_data(n_folds=10, protein=config["protein"], fingerprint=config["fingerprint"])
    config.update(config_from_data)
    config["C"] =[10**i for i in range(-5,6)]
    config["gamma"] = [10**i for i in range(-14,0)]
    logger = get_exp_logger(config)

    ### Prepare experiment ###
    E = {"config": config, "experiments":[]}

    def fit_svm(config):
        ### Prepare result holders ###b
        values = {}
        results = {}
        monitors = {}
        E = {"config": config, "results": results, "monitors":monitors, "values":values}

        ### Print experiment header ###
        print_exp_name(config)

        ### Train ###
        monitors["acc_fold"] = []
        monitors["mcc_fold"] = []
        monitors["wac_fold"] = []
        monitors["cm"] = [] # confusion matrix

        results["mean_acc"] = 0
        results["mean_mcc"] = 0


        X, Y = D["X"], D["Y"]
        values["mean_cls"] = Y.mean()
        values["transformers"] = []

        for fold in D["folds"]:
            tr_id, ts_id = fold["train_id"], fold["test_id"]

            X_train, Y_train, X_test, Y_test = X[tr_id], Y[tr_id], X[ts_id], Y[ts_id]

            min_max_scaler = MinMaxScaler()
            X_train = min_max_scaler.fit_transform(X_train.todense())
            X_test = min_max_scaler.transform(X_test.todense())

            values["transformers"].append(min_max_scaler) # Just in case

            m = SVC(C=config["C"], gamma=config["gamma"], class_weight="auto")

            m.fit(X_train, Y_train)
            Y_pred = m.predict(X_test)
            acc_fold, mcc_fold = accuracy_score(Y_test, Y_pred), matthews_corrcoef(Y_test, Y_pred)
            cm = confusion_matrix(Y_test, Y_pred)
            tp, fn, fp, tn = cm[1,1], cm[1,0], cm[0,1], cm[0,0]

            monitors["cm"].append(cm)
            monitors["wac_fold"].append(0.5*tp/float(tp+fn) + 0.5*tn/float(tn+fp))
            monitors["acc_fold"].append(acc_fold)
            monitors["mcc_fold"].append(mcc_fold)

        monitors["acc_fold"] = np.array(monitors["acc_fold"])
        monitors["mcc_fold"] = np.array(monitors["mcc_fold"])
        monitors["wac_fold"] = np.array(monitors["wac_fold"])

        results["mean_acc"] = monitors["acc_fold"].mean()
        results["mean_mcc"] = monitors["mcc_fold"].mean()
        results["mean_wac"] = monitors["wac_fold"].mean()

        logger.info(results)

        return E


    cv_configs = generate_configs(config, ["C", "gamma"])
    for c in cv_configs:
        E["experiments"].append(fit_svm(c))

    save_exp(E)

    best_e = E["experiments"][0]
    for e in E["experiments"]:
        if e["results"]["mean_mcc"] > best_e["results"]["mean_mcc"]:
            best_e = e
    logger.info(best_e)
    logger.info("Done")

if __name__ == "__main__":
    fit_svms()