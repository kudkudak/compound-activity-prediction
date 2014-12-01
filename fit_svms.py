from misc.utils import *
from misc.experiment_utils import get_exp_options, print_exp_header, \
    save_exp, get_exp_logger, generate_configs, print_exp_name
from data_api import prepare_experiment_data, prepare_experiment_data_embedded, get_raw_training_data, compute_jaccard_kernel


from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.kernel_approximation import Nystroem
from sklearn import datasets, svm, pipeline
from sklearn.preprocessing import MinMaxScaler

def fit_svms(config_in = None):
    #### Load config and data ####
    config = {"protein":0, "fingerprint":4,"n_folds":10, "kernel":"rbf", \
              "use_embedding": 0, "K":20, "max_hashes":1000, "seed":0, "C_min":-5, "C_max":6, "gamma_min":-14, "gamma_max":0}

    if config_in is None:
        config.update(get_exp_options(config))
    else:
        config.update(config_in)

    print_exp_header(config)

    if config["use_embedding"] == 0:
        D, config_from_data = prepare_experiment_data(n_folds=10, seed=config["seed"], \
                                                      protein=config["protein"], fingerprint=config["fingerprint"], force_reload=True)
    else:
        D, config_from_data = prepare_experiment_data_embedded(n_folds=10, seed=config["seed"], K=config["K"], \
                                                      max_hashes=config["max_hashes"],
                                                      protein=config["protein"], fingerprint=config["fingerprint"])

    X, Y = get_raw_training_data(protein = config["protein"], fingerprint=config["fingerprint"], force_reload=True)


    config.update(config_from_data)

    if config["kernel"] != "linear":
        config["C"] =[10**i for i in range(config["C_min"],1+config["C_max"])]
    else:
        config["C"] =[10**(i/float(2)) for i in range(2*config["C_min"],2*(1+config["C_max"]))]


    config["gamma"] = [10**i for i in range(config["gamma_min"],config["gamma_max"]+1)]
    logger = get_exp_logger(config)

    ### Prepare experiment ###
    E = {"config": config, "experiments":[]}
	   
    if config["kernel"] == "jaccard":
    	K = compute_jaccard_kernel(protein=config["protein"], fingerprint=config["fingerprint"], seed=config["seed"])

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
        monitors["n_support"] = []
        monitors["train_time"] = []
        monitors["test_time"] = []
        monitors["cm"] = [] # confusion matrix

        results["mean_acc"] = 0
        results["mean_mcc"] = 0


        values["mean_cls"] = Y.mean()
        values["transformers"] = []


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

            if config["kernel"] == "rbf":
                m = SVC(C=config["C"], gamma=config["gamma"], class_weight="auto")
                clf = m
            elif config["kernel"] == "linear":
                m = SVC(C=config["C"], kernel="linear", class_weight="auto")
                clf = m
            elif config["kernel"] == "jaccard":
                m = SVC(C=config["C"], kernel="precomputed", class_weight="auto")
                clf = m
            elif config["kernel"] == "rbf-nystroem":
                feature_map_nystroem = Nystroem(gamma=config["gamma"], random_state=config["seed"])
                clf = SVC(kernel="linear", C=config["C"] , class_weight="auto")
                m = pipeline.Pipeline([("feature_map", feature_map_nystroem),
                                                    ("svm", clf)])
            tstart = time.time()
            if config["kernel"] == "jaccard":
                m.fit(K[tr_id,:][:, tr_id], Y_train)
            else:
                m.fit(X_train, Y_train)

            monitors["train_time"].append(time.time() - tstart)
            monitors["n_support"].append(clf.n_support_)
            tstart = time.time()
            if config["kernel"] == "jaccard":
                Y_pred = clf.predict(K[tst_id, :][:, tr_id])
            else:
                Y_pred = m.predict(X_test)

            monitors["test_time"].append(time.time() - tstart)

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

    if config["kernel"] == "linear":
        cv_configs = generate_configs(config, ["C"])
    else:
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
