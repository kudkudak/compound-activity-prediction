from misc.utils import *
from misc.experiment_utils import get_exp_options, print_exp_header, \
    save_exp, get_exp_logger, generate_configs, print_exp_name
from data_api import prepare_experiment_data, prepare_experiment_data_embedded
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
from gmum.melc import *
from sklearn.preprocessing import MinMaxScaler

def fit_melcs(config_in = None):
    #### Load config and data ####
    config = {"protein":0, "fingerprint":4,"n_folds":10,
              "use_embedding": 0, "K":15, "max_hashes":300, "seed":0}

    if config_in is None:
        config.update(get_exp_options(config))
    else:
        config.update(config_in)

    print_exp_header(config)

    if config["use_embedding"] == 0:
        D, config_from_data = prepare_experiment_data(n_folds=10, seed=config["seed"], \
                                                      protein=config["protein"], fingerprint=config["fingerprint"])
    else:
        D, config_from_data = prepare_experiment_data_embedded(n_folds=10, seed=config["seed"], K=config["K"], \
                                                      max_hashes=config["max_hashes"],
                                                      protein=config["protein"], fingerprint=config["fingerprint"])

    config.update(config_from_data)
    config["gamma"] = np.linspace(0.1, 1.5, 15)
    logger = get_exp_logger(config)

    ### Prepare experiment ###
    E = {"config": config, "experiments":[]}

    def fit_melc(config):
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
        values["models"] = []

        for fold in D["folds"]:
            if config["use_embedding"] == 0:
                tr_id, ts_id = fold["train_id"], fold["test_id"]
                X_train, Y_train, X_test, Y_test = X[tr_id], Y[tr_id], X[ts_id], Y[ts_id]
                min_max_scaler = MinMaxScaler()
                X_train = min_max_scaler.fit_transform(X_train.todense())
                X_test = min_max_scaler.transform(X_test.todense())
            else:
                X_train, Y_train, X_test, Y_test = fold["X_train"], fold["Y_train"], fold["X_test"], fold["Y_test"]
                min_max_scaler = MinMaxScaler()
                X_train = min_max_scaler.fit_transform(X_train)
                X_test = min_max_scaler.transform(X_test)


            values["transformers"].append(min_max_scaler) # Just in case

            clf = MELC(base_objective=DCS(gamma=config["gamma"]), method="L-BFGS-B", random_state=666,
                        n_iters=4, n_jobs=4, verbose=0, on_sphere=True)

            values["models"].append(clf)

            clf.fit(X_train, Y_train)
            Y_pred = clf.predict(X_test)
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


    cv_configs = generate_configs(config, ["gamma"])

    for c in cv_configs:
        E["experiments"].append(fit_melc(c))

    save_exp(E)

    best_e = E["experiments"][0]
    for e in E["experiments"]:
        if e["results"]["mean_mcc"] > best_e["results"]["mean_mcc"]:
            best_e = e
    logger.info(best_e)
    logger.info("Done")

if __name__ == "__main__":
    fit_melcs()