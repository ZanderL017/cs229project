import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(mode="every", log=False):
    """
    Loads in all datasets, returned in a dictionary, unless mode is specificed to be a single dataset,
    in which case a dictionary with only that dataset is returned.
    """
    dsets = {}
    if mode == "every":
        dsets["read"] = pd.read_csv("data/READCopyProtein50.csv")
        dsets["coad"] = pd.read_csv("data/COADCopyProtein50.csv")
        dsets["gse"] = pd.read_csv("data/GSE62254CopyConvertedProtein.csv")
        dsets["all"] = pd.concat((dsets["read"], dsets["coad"], dsets["gse"]))
    elif mode == "read":
        dsets["read"] = pd.read_csv("data/READCopyProtein50.csv")
    elif mode == "coad":
        dsets["coad"] = pd.read_csv("data/COADCopyProtein50.csv")
    elif mode == "gse":
        dsets["gse"] = pd.read_csv("data/GSE62254CopyConvertedProtein.csv")
    elif mode == "all":  
        dsets["all"] = pd.concat((dsets["read"], dsets["coad"], dsets["gse"]))
    else:
        raise Exception("Invalid mode. Valid modes are read, coad, gse, all, every.")
        
    for dname in dsets:
        dsets[dname] = clean_dataset(dsets[dname], dname, log)

    return dsets

def clean_dataset(df, name, log):
    """
    Removes unneccasry columns, drops null values, and give log-scaled data if log is true.
    """
    df.drop("Unnamed: 0", axis=1, inplace=True)
    df.dropna(axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    labels = ["COVAR_M", "COVAR_N_status"]
    dfX = df.drop(labels, axis=1)
    if name == "gse":
        dfX = dfX.apply(np.exp)
    if log:
        dfX = dfX.apply(np.log)
    df = dfX.join(df[labels])
    return df

def split_dataset(df):
    """
    Splits a dataframe into train and testing input and outputs dataframes.
    """
    classes = {
        "00": 0,
        "01": 1,
        "10": 2,
        "11": 3,
    }
    X = df.drop(["COVAR_M", "COVAR_N_status"], axis=1)
    y = df[["COVAR_M", "COVAR_N_status"]]
    combined_class = y.astype(str).agg("".join, axis=1).map(classes)
    y.insert(2, "Combined", combined_class)
    return train_test_split(X, y, test_size=0.33, random_state=17000)