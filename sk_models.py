from sklearn import metrics, linear_model, ensemble, neural_network

def run_sk(model_type, data_splits, f1=False):
    X_train, X_test, y_train, y_test = data_splits
    model = model_type
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results = {}
    results["accuracy"] = metrics.accuracy_score(y_test, preds)
    if f1:
        results["f1"] = metrics.f1_score(y_test, preds)
    return results

def lr_baseline(data_splits, f1):
    model = linear_model.LogisticRegression(max_iter=1000)
    return run_sk(model, data_splits, f1)

def lr_elastic(data_splits, f1):
    model = linear_model.LogisticRegression(max_iter=1000, penalty="elasticnet", solver="saga", l1_ratio=0.5)
    return run_sk(model, data_splits, f1)

def basic_nn(data_splits, f1):
    model = neural_network.MLPClassifier(50, max_iter=2000)
    return run_sk(model, data_splits, f1)

def random_forest(data_splits, f1):
    model = ensemble.RandomForestClassifier()
    return run_sk(model, data_splits, f1)