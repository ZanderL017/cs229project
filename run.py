from pyexpat import model
import pandas as pd
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import load_data, split_dataset, split, add_combined, scale_dataset
from processing import extract_by_vars, extract_by_means
from sk_models import lr_baseline, lr_elastic, random_forest, basic_nn
from sklearn import model_selection, decomposition, manifold, preprocessing
from run_igtd import run_igtd



np.random.seed(17000)

parser = argparse.ArgumentParser(description="Train and evaluate")
parser.add_argument("-d", "--dataset", default="read", type=str)
parser.add_argument("-f", "--feature-extractor", default="", type=str)
parser.add_argument("-l", "--log-transform", default=1, type=int)
parser.add_argument("-v", "--vae-csv", default="", type=str)
parser.add_argument("-k", "--top-k", default=1024, type=int)

def main():
    args = parser.parse_args()
    dname = args.dataset
    feature_extractor = args.feature_extractor
    log = bool(args.log_transform)
    vae_csv = args.vae_csv
    top_k = args.top_k
    
    data = load_data(mode=dname, log=log)
    df = data[dname]
    df = add_combined(df)

    print(dname.upper(), "loaded.")

    train_i, test_i = model_selection.train_test_split(np.arange(len(df)), test_size=0.3, random_state=17)
    labels = ["COVAR_M", "COVAR_N_status", "Combined"]
    data_splits = {}

    # none
    if feature_extractor == "":
        X = df.drop(labels + ["sample"], axis=1)
        X = preprocessing.StandardScaler().fit_transform(X)
        y = df[labels]
        for label in labels:
            data_splits[label] = split(X, y[label], train_i, test_i)

    # means
    if feature_extractor == "mean":
        X = df.drop(labels + ["sample"], axis=1)
        y = df[labels]
        labels.remove("Combined")
        for label in labels:
            features = extract_by_means(X.loc[train_i], y.loc[train_i], label, 1, top_k)
            data_splits[label] = split(X[features], y[label], train_i, test_i)

    # vars
    if feature_extractor == "var":
        X = df.drop(labels + ["sample"], axis=1)
        y = df[labels]
        labels.remove("Combined")
        for label in labels:
            features = extract_by_vars(X.loc[train_i], y.loc[train_i], label, 1, top_k)
            data_splits[label] = split(X[features], y[label], train_i, test_i)
    # vae
    if feature_extractor == "vae":
        assert vae_csv, "Make sure to specify a csv to use for vae features"
        df_vae = pd.read_csv(vae_csv)
        assert (df_vae["sample"].values == df["sample"].values).all(), "Make sure the csv corresponds to the current dataset"
        df_vae = df_vae.join(df[labels])
        X = df_vae.drop(labels + ["sample"], axis=1)
        y = df_vae[labels]
        for label in labels:
            data_splits[label] = split(X, y[label], train_i, test_i)
    # pca
    if feature_extractor == "pca":
        X = df.drop(labels + ["sample"], axis=1)
        y = df[labels]
        pca = decomposition.PCA()
        pca.fit(X.loc[train_i])
        for label in labels:
            data_splits[label] = split(pca.transform(X), y[label], train_i, test_i)
    
    # tsne
    if feature_extractor == "tsne":
        X = df.drop(labels + ["sample"], axis=1)
        X = preprocessing.StandardScaler().fit_transform(X)
        y = df[labels]
        tsne = manifold.TSNE()
        tsne.fit(X.loc[train_i])
        for label in labels:
            data_splits[label] = split(tsne.transform(X), y[label], train_i, test_i)
    # umap
    
    # Run models
    accs = {}
    f1s = {}
    print("Running Models...")
    for label in labels:
        data_split = data_splits[label]
        f1 = False if label == "Combined" else True
        baseline = lr_baseline(data_split, f1)
        elastic = lr_elastic(data_split, f1)
        rf = random_forest(data_split, f1)
        nn = basic_nn(data_split, f1)
        accs[(label, "lr_baseline")] = baseline["accuracy"]
        accs[(label, "lr_elastic")] = elastic["accuracy"]
        accs[(label, "rf_acc")] = rf["accuracy"]
        accs[(label, "nn_acc")] = nn["accuracy"]
        if label != "Combined":
            f1s[(label, "lr_baseline")] = baseline["f1"]
            f1s[(label, "lr_elastic")] = elastic["f1"]
            f1s[(label, "rf_acc")] = rf["f1"]
            f1s[(label, "nn_acc")] = nn["f1"]
        print("Finished", label)
    #run cnn model
    if feature_extractor in ["var", "mean", "vae"]:
        results = run_igtd(data_splits, labels)
        for label in results:
            accs[(label, "cnn")] = results[label]

    # print results
    for key in accs.keys():
        print(f'{"Accuracy for " + key[0] + " on " + key[1]:42}', "===" , round(accs[key], 4))
    print()
    for key in f1s.keys():
        print(f'{"F1 for " + key[0] + " on " + key[1]:42}', "===" , round(f1s[key], 4))
    


if __name__ == "__main__":
    main()

