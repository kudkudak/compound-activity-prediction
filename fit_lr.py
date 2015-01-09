from misc.utils import *

from misc.experiment_utils import get_exp_options, print_exp_header, \
    save_exp, get_exp_logger, generate_configs, print_exp_name
from data_api import prepare_experiment_data, prepare_experiment_data_embedded, get_raw_training_data
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model


def fit_lrs(config_in = None):
    #### Load config and data ####
    config = {"protein":0, "fingerprint":4,"n_folds":10,
              "use_embedding": 1, "K":20, "max_hashes":1000, "seed":0,  "C_min":-3, "C_max":7}

    if config_in is None:
        config.update(get_exp_options(config))
    else:
        config.update(config_in)


    D, config_from_data = prepare_experiment_data_embedded(n_folds=10, seed=config["seed"], K=config["K"], \
                                                  max_hashes=config["max_hashes"],
                                                  protein=config["protein"], fingerprint=config["fingerprint"])


    config.update(config_from_data)
    config["C"] = [10.0**(i/float(2)) for i in range(2*config["C_min"],2*(1+config["C_max"]))]

    print config["C"]

    logger = get_exp_logger(config)

    ### Prepare experiment ###
    E = {"config": config, "experiments":[]}

    def fit_lr(config):
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
        monitors["clf"] = []
        monitors["train_time"] = []
        monitors["test_time"] = []

        results["mean_acc"] = 0
        results["mean_mcc"] = 0


        values["transformers"] = []

        for fold in D["folds"]:
            X_train, Y_train, X_test, Y_test = fold["X_train"], fold["Y_train"], fold["X_test"], fold["Y_test"]
            min_max_scaler = MinMaxScaler()
            X_train = min_max_scaler.fit_transform(X_train)
            X_test = min_max_scaler.transform(X_test)

            clf =sklearn.linear_model.LogisticRegression (C=config["C"], class_weight="auto")
            tstart = time.time()

            monitors["train_time"].append(time.time() - tstart)
            clf.fit(X_train.astype(float), Y_train.astype(float).reshape(-1))
            tstart = time.time()
            Y_pred = clf.predict(X_test.astype(float))
            monitors["test_time"].append(time.time() - tstart)

            acc_fold, mcc_fold = accuracy_score(Y_test, Y_pred), matthews_corrcoef(Y_test, Y_pred)
            cm = confusion_matrix(Y_test, Y_pred)
            tp, fn, fp, tn = cm[1,1], cm[1,0], cm[0,1], cm[0,0]

            monitors["clf"].append(clf)
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

    cv_configs = generate_configs(config, ["C"])

    for c in cv_configs:
        print c
        E["experiments"].append(fit_lr(c))

    save_exp(E)

    best_e = E["experiments"][0]
    for e in E["experiments"]:
        if e["results"]["mean_wac"] > best_e["results"]["mean_wac"]:
            best_e = e
    logger.info(best_e)
    logger.info("Done")

if __name__ == "__main__":
    fit_lrs()
