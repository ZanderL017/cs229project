import numpy as np
import pandas as pd

def scale(s):
    return (s - s.min()) / (s.max() - s.min())

def normalize(s):
    return (s - s.mean()) / s.std()

def extract_by_means(X_train, y_train, label, class_i, top_k):
    diffs = abs(X_train[y_train[label] == class_i].mean() - X_train.mean())
    sorted_diffs = diffs.sort_values(ascending=False).index.to_numpy()
    return sorted_diffs[:top_k]

def extract_by_vars(X_train, y_train, label, class_i, top_k):
    diffs = abs(X_train[y_train[label] == class_i].var() - X_train.var())
    sorted_diffs = diffs.sort_values(ascending=False).index.to_numpy()
    return sorted_diffs[:top_k]