# %%
import numpy as np
import time
from CRR import ConformalRidgeRegressor
from Ridge_point_predictor import OnlineRidgeRegressor, ReallyStupidPredictor
import pickle
from tqdm import tqdm

# Target error rate and step size
epsilon = 0.1
# gamma = 0.005

# Guarantee of ACI is absolute deviation of error rate <= bound

tot_init = time.time()

# Set the number of runs
N = 1000

experiment = {}

for j, gamma in tqdm(enumerate([0, 0.001, 0.005, 0.007, 0.01]), total=5):
    time.sleep(3)
    tot_init = time.time()
    # Test 1: IID
    for seed in tqdm(range(N), total=N, desc=f'Running linear experiment: gamma = {gamma}'):
        rnd_gen = np.random.default_rng(seed)


        # Generate data
        X = rnd_gen.normal(loc=0, scale=1, size=(2000, 4))
        beta = np.array([2, 1, 0, 0])
        Y = X @ beta + rnd_gen.normal(loc=0, scale=1, size=2000)

        # Split data
        initial_training_size = 100
        X_train = X[:initial_training_size]
        y_train = Y[:initial_training_size]

        X_run = X[initial_training_size:]
        y_run = Y[initial_training_size:]

        try:
            aci_bound = (max(epsilon, 1-epsilon) + gamma) / (gamma * y_run.shape[0])
        except ZeroDivisionError:
            aci_bound = np.nan
        experiment[seed] = {'iid': {}, 'change_points': {}, 'drift': {}, 'aci_bound': aci_bound, 'epsilon': epsilon, 'gamma': gamma}

        # Run full CP
        cp = ConformalRidgeRegressor(a=0, warnings=False)

        time_init_cp = time.time()
        cp.learn_initial_training_set(X_train, y_train)

        eps = epsilon


        res = np.empty((X_run.shape[0], 3))
        for i, (obj, label) in enumerate(zip(X_run, y_run)):
            # Reality presents the object x
            x = obj
            # Forecaster outputs Gamma
            Gamma, precomputed = cp.predict(x, epsilon=eps, bounds='both', return_update=True)
            width = cp.width(Gamma)
            # Reality presents the label
            y = label
            cp.learn_one(x, y, precomputed)
            err = cp.err(Gamma, label)

            eps += gamma*(epsilon - err)
            #eps = max(2/cp.X.shape[0], eps)

            res[i, 0] = err
            res[i, 1] = width
            res[i, 2] = eps

        time_cp = time.time() - time_init_cp

        experiment[seed]['iid']['cp'] = {'time': time_cp, 'result': res, 'epsilon': epsilon}


        # Run OLS
        ols = OnlineRidgeRegressor(a=0)

        time_init_ols = time.time()

        ols.learn_initial_training_set(X_train, y_train)

        eps = epsilon

        res = np.empty((X_run.shape[0], 3))
        for i, (obj, label) in enumerate(zip(X_run, y_run)):
            # Reality presents the object x
            x = obj
            Gamma = ols.predict_interval(x, epsilon=eps)
            width = ols.width(Gamma)
            # Reality reveals y
            y = label
            ols.learn_one(x, y)
            err = ols.err(Gamma, label)

            eps += gamma*(epsilon - err)
            #eps = max(2/ols.X.shape[0], eps)

            res[i, 0] = err
            res[i, 1] = width
            res[i, 2] = eps

        time_ols = time.time() - time_init_ols

        experiment[seed]['iid']['ols'] = {'time': time_ols, 'result': res, 'epsilon': epsilon}

        # Run Stupid
        stupid = ReallyStupidPredictor(low=-1, high=1)

        time_init_stupid = time.time()

        eps = epsilon

        res = np.empty((X_run.shape[0], 3))
        for i, (obj, label) in enumerate(zip(X_run, y_run)):
            # Reality presents the object x
            x = obj
            Gamma = stupid.predict_interval(x, epsilon=eps)
            width = stupid.width(Gamma)
            # Reality reveals y
            y = label
            err = stupid.err(Gamma, label)

            eps += gamma*(epsilon - err)
            #eps = max(2/ols.X.shape[0], eps)

            res[i, 0] = err
            res[i, 1] = width
            res[i, 2] = eps

        time_stupid = time.time() - time_init_stupid

        experiment[seed]['iid']['stupid'] = {'time': time_stupid, 'result': res, 'epsilon': epsilon}


    # Test 2: Change points

        beta1 = np.array([2, 1, 0, 0])
        beta2 = np.array([0, -2,-1, 0])
        beta3 = np.array([0, 0, 2, 1])

        Y1 = X[:500] @ beta1 + rnd_gen.normal(loc=0, scale=1, size=500)
        Y2 = X[500:1500] @ beta2 + rnd_gen.normal(loc=0, scale=1, size=1000)
        Y3 = X[1500:] @ beta3 + rnd_gen.normal(loc=0, scale=1, size=500)
        Y = np.concatenate([Y1, Y2, Y3])

        # Split data
        initial_training_size = 100
        X_train = X[:initial_training_size]
        y_train = Y[:initial_training_size]

        X_run = X[initial_training_size:]
        y_run = Y[initial_training_size:]

        # Run full CP
        cp = ConformalRidgeRegressor(a=0, warnings=False)

        time_init_cp = time.time()
        cp.learn_initial_training_set(X_train, y_train)

        eps = epsilon


        res = np.empty((X_run.shape[0], 3))
        for i, (obj, label) in enumerate(zip(X_run, y_run)):
            # Reality presents the object x
            x = obj
            # Forecaster outputs Gamma
            Gamma, precomputed = cp.predict(x, epsilon=eps, bounds='both', return_update=True)
            width = cp.width(Gamma)
            # Reality presents the label
            y = label
            cp.learn_one(x, y, precomputed)
            err = cp.err(Gamma, label)

            eps += gamma*(epsilon - err)
            #eps = max(2/cp.X.shape[0], eps)

            res[i, 0] = err
            res[i, 1] = width
            res[i, 2] = eps

        time_cp = time.time() - time_init_cp

        experiment[seed]['change_points']['cp'] = {'time': time_cp, 'result': res, 'epsilon': epsilon}


        # Run OLS
        ols = OnlineRidgeRegressor(a=0)

        time_init_ols = time.time()

        ols.learn_initial_training_set(X_train, y_train)

        eps = epsilon

        res = np.empty((X_run.shape[0], 3))
        for i, (obj, label) in enumerate(zip(X_run, y_run)):
            # Reality presents the object x
            x = obj
            Gamma = ols.predict_interval(x, epsilon=eps)
            width = ols.width(Gamma)
            # Reality reveals y
            y = label
            ols.learn_one(x, y)
            err = ols.err(Gamma, label)

            eps += gamma*(epsilon - err)
            #eps = max(2/ols.X.shape[0], eps)

            res[i, 0] = err
            res[i, 1] = width
            res[i, 2] = eps

        time_ols = time.time() - time_init_ols

        experiment[seed]['change_points']['ols'] = {'time': time_ols, 'result': res, 'epsilon': epsilon}

        # Run Stupid
        stupid = ReallyStupidPredictor(low=-1, high=1)

        time_init_stupid = time.time()

        eps = epsilon

        res = np.empty((X_run.shape[0], 3))
        for i, (obj, label) in enumerate(zip(X_run, y_run)):
            # Reality presents the object x
            x = obj
            Gamma = stupid.predict_interval(x, epsilon=eps)
            width = stupid.width(Gamma)
            # Reality reveals y
            y = label
            err = stupid.err(Gamma, label)

            eps += gamma*(epsilon - err)
            #eps = max(2/ols.X.shape[0], eps)

            res[i, 0] = err
            res[i, 1] = width
            res[i, 2] = eps

        time_stupid = time.time() - time_init_stupid

        experiment[seed]['change_points']['stupid'] = {'time': time_stupid, 'result': res, 'epsilon': epsilon}

    # Test 3: Drifring mean

        beta1 = np.array([2, 1, 0, 0])
        betaN = np.array([0, 0, 2, 1])
        Y = np.zeros((X.shape[0]))
        for i, x in enumerate(X):
            Y[i] = x @ ((1 - i/(X.shape[0]-1))*beta1 + (i/(X.shape[0]-1))*betaN) + rnd_gen.normal(loc=0, scale=1)

        # Split data
        initial_training_size = 100
        X_train = X[:initial_training_size]
        y_train = Y[:initial_training_size]

        X_run = X[initial_training_size:]
        y_run = Y[initial_training_size:]

        # Run full CP
        cp = ConformalRidgeRegressor(a=0, warnings=False)

        time_init_cp = time.time()
        cp.learn_initial_training_set(X_train, y_train)

        eps = epsilon


        res = np.empty((X_run.shape[0], 3))
        for i, (obj, label) in enumerate(zip(X_run, y_run)):
            # Reality presents the object x
            x = obj
            # Forecaster outputs Gamma
            Gamma, precomputed = cp.predict(x, epsilon=eps, bounds='both', return_update=True)
            width = cp.width(Gamma)
            # Reality presents the label
            y = label
            cp.learn_one(x, y, precomputed)
            err = cp.err(Gamma, label)

            eps += gamma*(epsilon - err)
            #eps = max(2/cp.X.shape[0], eps)

            res[i, 0] = err
            res[i, 1] = width
            res[i, 2] = eps

        time_cp = time.time() - time_init_cp

        experiment[seed]['drift']['cp'] = {'time': time_cp, 'result': res, 'epsilon': epsilon}


        # Run OLS
        ols = OnlineRidgeRegressor(a=0)

        time_init_ols = time.time()

        ols.learn_initial_training_set(X_train, y_train)

        eps = epsilon

        res = np.empty((X_run.shape[0], 3))
        for i, (obj, label) in enumerate(zip(X_run, y_run)):
            # Reality presents the object x
            x = obj
            Gamma = ols.predict_interval(x, epsilon=eps)
            width = ols.width(Gamma)
            # Reality reveals y
            y = label
            ols.learn_one(x, y)
            err = ols.err(Gamma, label)

            eps += gamma*(epsilon - err)
            #eps = max(2/ols.X.shape[0], eps)

            res[i, 0] = err
            res[i, 1] = width
            res[i, 2] = eps

        time_ols = time.time() - time_init_ols

        experiment[seed]['drift']['ols'] = {'time': time_ols, 'result': res, 'epsilon': epsilon}

        # Run Stupid
        stupid = ReallyStupidPredictor(low=-1, high=1)

        time_init_stupid = time.time()

        eps = epsilon

        res = np.empty((X_run.shape[0], 3))
        for i, (obj, label) in enumerate(zip(X_run, y_run)):
            # Reality presents the object x
            x = obj
            Gamma = stupid.predict_interval(x, epsilon=eps)
            width = stupid.width(Gamma)
            # Reality reveals y
            y = label
            err = stupid.err(Gamma, label)

            eps += gamma*(epsilon - err)
            #eps = max(2/ols.X.shape[0], eps)

            res[i, 0] = err
            res[i, 1] = width
            res[i, 2] = eps

        time_stupid = time.time() - time_init_stupid

        experiment[seed]['drift']['stupid'] = {'time': time_stupid, 'result': res, 'epsilon': epsilon}
    # %%

    with open(f'ACI_linear_experiment_{N}_{j}.pkl', 'wb') as fp:
        pickle.dump(experiment, fp)
    # %%
    with open(f'ACI_linear_experiment_{N}_{j}.pkl', 'rb') as fp:
        test_load = pickle.load(fp)
    # %%
    print(f'Mean time: {(time.time() - tot_init)/N:,.4f} s')