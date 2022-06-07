import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def load_data(mode="every", log=False):
    """
    Loads in all datasets, returned in a dictionary, unless mode is specificed to be a single dataset,
    in which case a dictionary with only that dataset is returned.
    """
    dsets = {}
    if mode == "read" or mode == "all":
        dsets["read"] = clean_dataset(pd.read_csv("data/READCopyProtein50.csv"), "read", log)
    if mode == "coad" or mode == "all":
        dsets["coad"] = clean_dataset(pd.read_csv("data/COADCopyProtein50.csv"), "coad", log)
    if mode == "gse" or mode == "all":
        dsets["gse"] = clean_dataset(pd.read_csv("data/GSE62254CopyConvertedProtein.csv"), "gse", log)
    if mode == "all":  
        dsets["all"] = pd.concat((dsets["read"], dsets["coad"], dsets["gse"]))
    if mode not in ["read", "coad", "gse", "all"]:
        raise Exception("Invalid mode. Valid modes are read, coad, gse, all.")

    return dsets

def clean_dataset(df, name, log):
    """
    Removes unneccasry columns, drops null values, and give log-scaled data if log is true.
    """
    df.rename({"Unnamed: 0": "sample"}, axis=1, inplace=True)
    df.dropna(axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    labels = ["COVAR_M", "COVAR_N_status", "sample"]
    dfX = df.drop(labels, axis=1)
    if name == "gse":
        dfX = dfX.apply(np.exp)
    if log:
        dfX = dfX.apply(np.log)
    df = dfX.join(df[labels])
    return df

def add_combined(df):
    """
    Adds a column to the dataframe to create a multiclass label.
    """
    classes = {
        "00": 0,
        "01": 1,
        "10": 2,
        "11": 3,
    }
    y = df[["COVAR_M", "COVAR_N_status"]]
    combined_class = y.astype(str).agg("".join, axis=1).map(classes)
    df.insert(0, "Combined", combined_class)
    return df

def split_dataset(df, random_state=17000):
    """
    Splits a dataframe into train and testing input and outputs dataframes.
    """
    df = add_combined(df)
    X = df.drop(["COVAR_M", "COVAR_N_status", "sample", "Combined"], axis=1)
    y = df[["COVAR_M", "COVAR_N_status", "Combined"]]
    return train_test_split(X, y, test_size=0.33, random_state=random_state)

def scale_dataset(df):
    X = df.drop(["COVAR_M", "COVAR_N_status", "sample", "Combined"], axis=1)
    y = df[["COVAR_M", "COVAR_N_status", "Combined"]]
    X = preprocessing.StandardScaler().fit_transform(X)
    return X.join(y)

def split(X, y, train_i, test_i):
    if type(X) != np.ndarray:
        X = X.to_numpy()
    if type(y) != np.ndarray:
        y = y.to_numpy()
    X_train = X[train_i]
    X_test = X[test_i]
    y_train = y[train_i]
    y_test = y[test_i]
    return X_train, X_test, y_train, y_test