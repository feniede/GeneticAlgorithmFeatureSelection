from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import json

def gen_partitions(X_full, Y_full, n_splits=5, n_repeats=5, seed=1234):
    
    X_trains = []
    X_tests = []
    Y_trains = []
    Y_tests = []
    X_trains_1 = []
    X_vals = []
    Y_trains_1 = []
    Y_vals = []

    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for i, (train_index, test_index) in enumerate(rkf.split(X_full, Y_full)):
        
        X_train = X_full[train_index]
        X_test = X_full[test_index]
        Y_train = Y_full[train_index]
        Y_test = Y_full[test_index]
        
        scaler = MinMaxScaler() #esto es para escalar.
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        X_trains.append(X_train)
        X_tests.append(X_test)
        Y_trains.append(Y_train)
        Y_tests.append(Y_test)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
        for j, (train_1_index, val_index) in enumerate(skf.split(X_train, Y_train)):
            X_train_1 = X_train[train_1_index]
            X_val = X_train[val_index]
            Y_train_1 = Y_train[train_1_index]
            Y_val = Y_train[val_index]
            
            X_trains_1.append(X_train_1)
            X_vals.append(X_val)
            Y_trains_1.append(Y_train_1)
            Y_vals.append(Y_val)
            
    return X_trains, X_tests, X_trains_1, X_vals, Y_trains, Y_tests, Y_trains_1, Y_vals


def save_partitions_to_json(file_path, X_trains, X_tests, X_trains_1, X_vals, Y_trains, Y_tests, Y_trains_1, Y_vals):
    '''
    # Usage
    save_partitions_to_json("partitions_data.json", X_trains, X_tests, X_trains_1, X_vals, Y_trains, Y_tests, Y_trains_1, Y_vals)
    '''
    data = {
        "X_trains": [x.tolist() for x in X_trains],
        "X_tests": [x.tolist() for x in X_tests],
        "X_trains_1": [x.tolist() for x in X_trains_1],
        "X_vals": [x.tolist() for x in X_vals],
        "Y_trains": [y.tolist() for y in Y_trains],
        "Y_tests": [y.tolist() for y in Y_tests],
        "Y_trains_1": [y.tolist() for y in Y_trains_1],
        "Y_vals": [y.tolist() for y in Y_vals]
    }
    
    with open(file_path, 'w') as f:
        json.dump(data, f)
        
def load_partitions_from_json(file_path):
    ''' 
        # Usage
        X_trains, X_tests, X_trains_1, X_vals, Y_trains, Y_tests, Y_trains_1, Y_vals = load_partitions_from_json("partitions_data.json")
    '''
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    X_trains = [np.array(x) for x in data['X_trains']]
    X_tests = [np.array(x) for x in data['X_tests']]
    X_trains_1 = [np.array(x) for x in data['X_trains_1']]
    X_vals = [np.array(x) for x in data['X_vals']]
    Y_trains = [np.array(y) for y in data['Y_trains']]
    Y_tests = [np.array(y) for y in data['Y_tests']]
    Y_trains_1 = [np.array(y) for y in data['Y_trains_1']]
    Y_vals = [np.array(y) for y in data['Y_vals']]
    
    return X_trains, X_tests, X_trains_1, X_vals, Y_trains, Y_tests, Y_trains_1, Y_vals

if __name__ == '__main__':
    
    # Random seed
    seed = 1234
    
    # Loading database
    database_name = "database"
    sheet_name = "database"
    database = pd.read_excel(database_name + ".xlsx", sheet_name = sheet_name)
    
    # Preparing database
    data = database.values
    np.random.seed(seed=seed)
    idx = np.random.permutation(data.shape[0])
    data = data[idx,:]
    y = data[:,0].astype('int') 
    X = data[:,1:] 
    print(f"Total samples: {np.sum(y)}. Positive labels: {y.shape[0]}. Number of features: {X.shape[-1]}.")
    
    # Generating partitions
    X_trains, X_tests, X_trains_1, X_vals, Y_trains, Y_tests, Y_trains_1, Y_vals = gen_partitions(X, y, n_splits=5, n_repeats=5, seed=seed)
    
    # Saving partitions
    save_partitions_to_json("partitions_" + sheet_name + ".json", X_trains, X_tests, X_trains_1, X_vals, Y_trains, Y_tests, Y_trains_1, Y_vals)