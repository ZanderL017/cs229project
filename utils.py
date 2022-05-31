import pandas as pd

def load_data():
    read = pd.read_csv("data/READCopyProtein50.csv")
    coad = pd.read_csv("data/COADCopyProtein50.csv")
    gse = pd.read_csv("data/GSE62254CopyConvertedProtein.csv")
    all_data = pd.concat((read, coad, gse))
    
    for df in [read, coad, gse, all_data]:
        df.drop("Unnamed: 0", axis=1, inplace=True)
        df.dropna(axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)
        
    return read, coad, gse, all_data