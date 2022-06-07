import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from IGTD_functions import *
from data_utils import load_data, split_dataset
from processing import extract_by_vars, extract_by_means
from cnn_helper import IGTDImages, SimpleCNN

np.random.seed(17000)

def run_igtd(data_splits, labels, rows=32, cols=32):
    X_train, X_test, _, _ = data_splits["COVAR_M"]
    X = np.concatenate((X_train, X_test))
    #norm_d = min_max_transform(X)
    norm_d = X
    fea_dist_method = 'Euclidean'
    image_dist_method = 'Euclidean'
    ranking_feature, corr = generate_feature_distance_ranking(data=norm_d, method=fea_dist_method)
    coordinate, ranking_image = generate_matrix_distance_ranking(num_r=rows, num_c=cols, method=image_dist_method)
    index, err, time = IGTD(source=ranking_feature, target=ranking_image, save_folder=None, max_step=1500)
    min_id = np.argmin(err)
    data, samples = generate_image_data(data=norm_d, index=index[min_id, :], num_row=rows, num_column=cols, coord=coordinate, image_folder=None)
    assert sorted([int(s) for s in samples]) == [int(s) for s in samples]
    data = np.transpose(data, (2, 0, 1))
    X_train = data[:len(X_train)]
    X_test = data[len(X_train):]

    

    accs = {}
    for label in labels:
        _, _, y_train, y_test = data_splits[label]
        train_data = IGTDImages(X_train, y_train)
        test_data = IGTDImages(X_test, y_test)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)

        n_out = 4 if label == "Combined" else 2
        model = SimpleCNN(n_out)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.05)

        epochs = 30
        train_losses = []
        test_losses = []
        for epoch in range(epochs):
            train_loss = 0.0
            test_loss = 0.0
            model.train()
            for i, (image, target) in enumerate(train_loader):
                optimizer.zero_grad()
                out = model(image).squeeze()
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            model.eval()
            with torch.no_grad():
                for i, (image, target) in enumerate(test_loader):
                    out = model(image).squeeze()
                    loss = criterion(out, target)
                    
                    test_loss += loss.item()
                    
            train_loss /= len(train_loader)
            test_loss /= len(test_loader)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            print(train_loss, test_loss)

        accs[label] = accuracy(model, test_loader).item()
    return accs

def accuracy(model, *loaders):
    model.eval()
    with torch.no_grad():
        num_correct = 0
        tries = 0
        for loader in loaders:
            for i, (X, y) in enumerate(loader):
                out = model(X).squeeze()
                preds = torch.argmax(out, dim=1)
                num_correct += (preds == y).sum()
                tries += len(y)
        
        accuracy = num_correct / tries 
    return accuracy



