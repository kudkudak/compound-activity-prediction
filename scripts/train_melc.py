import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),".."))

from misc.utils import *
import scipy
from misc.lsh import *
from misc.experiment_utils import *
from data_api import *
from gmum.melc import *

D, config = prepare_experiment_data_embedded(protein=0, fingerprint=4, K=20, n_folds=10, seed=0, max_hashes=300)


clf = MELC(base_objective=DCS(gamma=1.5), method="L-BFGS_B", \
           random_state=666, n_iters=0, n_jobs=4, verbose=1, on_sphere=True)

from sklearn.preprocessing import MinMaxScaler
m = []
clfs = []
for f in [4]: # Train only on the problematic fold
    X_train, Y_train = D["folds"][f]["X_train"], D["folds"][f]["Y_train"]

    X_test, Y_test = D["folds"][f]["X_test"], D["folds"][f]["Y_test"]

    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)

    clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    clfs.append(clf)
    m.append(wac_score(Y_test, Y_pred))
print(m)

np.array(m).mean()