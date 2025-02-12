import numpy as np
rnd_gen = np.random.default_rng(2024)
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
from src.online_cp import ConformalNearestNeighboursClassifier
rnd_gen = np.random.default_rng(2024)

from joblib import Parallel, delayed
from tqdm import tqdm

plt.rcParams['svg.fonttype'] = 'none'
from matplotlib_inline.backend_inline import set_matplotlib_formats

# Set the desired output format
set_matplotlib_formats('svg')

from tol_colors import tol_cmap, tol_cset
cmap = tol_cset('bright')

import pandas as pd
import h5py

import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
from multiprocessing import Lock
lock = Lock()

def load_wine_original():
    wine_quality = fetch_ucirepo(id=186) 

    wine = wine_quality.data.original

    X = wine.drop(columns=['quality', 'color']).values
    Y = wine['quality'].astype('float').values

    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    label_space = np.unique(Y)

    return X, Y, label_space

def load_wine_change_point():
    wine_quality = fetch_ucirepo(id=186) 

    reds = wine_quality.data.original[wine_quality.data.original.color == 'red']
    whites = wine_quality.data.original[wine_quality.data.original.color == 'white']

    # Shuffle the sets (uncomment if we run all data)
    reds = reds.sample(reds.shape[0], random_state=2024)
    whites = whites.sample(whites.shape[0], random_state=2024)

    change_point = whites.shape[0]

    X_red = reds.drop(columns=['quality', 'color']).values
    X_white = whites.drop(columns=['quality', 'color']).values
    y_red = reds['quality'].astype('float').values
    y_white = whites['quality'].astype('float').values

    X = np.concatenate([X_white, X_red])
    Y = np.concatenate([y_white, y_red])

    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    label_space = np.unique(Y)

    return X, Y, label_space

def load_usps():
    path = 'data_martingale_experiment/usps.h5'
    with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        X_train = train.get('data')[:]
        y_train = train.get('target')[:]
        test = hf.get('test')
        X_test = test.get('data')[:]
        y_test = test.get('target')[:]

    X = np.concatenate([X_train, X_test])
    Y = np.concatenate([y_train, y_test])

    label_space = np.unique(Y)

    return X, Y, label_space

def load_satlog():
    # fetch dataset 
    statlog_landsat_satellite = fetch_ucirepo(id=146) 
    
    # data (as pandas dataframes) 
    X = statlog_landsat_satellite.data.features.values
    Y = statlog_landsat_satellite.data.targets.values.flatten()

    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    label_space = np.unique(Y)

    return X, Y, label_space

def load_dataset(name):
    if name == 'wine':
        return load_wine_original()
    elif name == 'usps':
        return load_usps()
    elif name == 'wine_change_point':
        return load_wine_change_point()
    elif name == 'satlog':
        return load_satlog()
    else:
        print(f'No dataset named {name}') 

        
def process_dataset(dataset, position):
    """Process a single dataset."""
    X, Y, label_space = load_dataset(dataset)

    sqrt_features = int(np.sqrt(X.shape[1]))
    k = sqrt_features - 1 if sqrt_features % 2 == 0 else sqrt_features

    cp = ConformalNearestNeighboursClassifier(k=k, label_space=label_space)

    epsilon = 0.1

    p_values_save = np.zeros(shape=(Y.shape[0], label_space.shape[0]))

    # Progress bar for this dataset
    with tqdm(total=Y.shape[0], desc=f'Running {dataset}', position=position, leave=True) as progress_bar:
        for i, (obj, lab) in enumerate(zip(X, Y)):
            # Make prediction
            Gamma, p_values, D = cp.predict(obj, epsilon=epsilon, return_p_values=True, return_update=True)

            p_values_save[i] = np.array([p_values[y] for y in label_space])

            # Learn the label
            cp.learn_one(obj, lab, D)

            # Update the progress bar
            progress_bar.update(1)

    columns = list(label_space)
    df_p_values = pd.DataFrame(p_values_save, columns=columns)
    df_p_values['label'] = Y

    output_path = f'data_protected_conformal_classification/p_values_{dataset}.csv'
    df_p_values.to_csv(output_path, index=False)

    res_str = f"Finished processing {dataset}. Results saved to {output_path}."
    return res_str


if __name__=='__main__':
    # List of datasets
    datasets = ['wine', 'usps', 'satlog', 'wine_change_point']

    # Parallel processing with joblib

    result = Parallel(n_jobs=-1, backend="loky")(delayed(process_dataset)(dataset, i) for i, dataset in enumerate(datasets))

    for s in result:
        print(s)