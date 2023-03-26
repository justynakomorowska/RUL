import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.float_format',  '{:,}'.format)

header = ['esn','cycles','opset1','opset2','TRA','T2','T24','T30','T50','P2','P15','P30','Nf',
           'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32', "blank1", "blank2"]

location1 = r'C:\Users\justyna.komorowska\Documents\___PROJEKTY___\PJATK\Magisterka\data\rawdata\train_FD001.txt'
location2 = r'C:\Users\justyna.komorowska\Documents\___PROJEKTY___\PJATK\Magisterka\data\rawdata\train_FD002.txt'
location3 = r'C:\Users\justyna.komorowska\Documents\___PROJEKTY___\PJATK\Magisterka\data\rawdata\train_FD003.txt'
location4 = r'C:\Users\justyna.komorowska\Documents\___PROJEKTY___\PJATK\Magisterka\data\rawdata\train_FD004.txt'

def read_in(file1, file2, file3, file4):
    df1 = pd.read_csv(r+f'{location1}', sep = " ", header = 0, names = header)
    df1['dataset'] = "FD001"
    df2 = pd.read_csv(r+f'{location2}', sep = " ", header = 0, names = header)
    df2['dataset'] = "FD002"
    df3 = pd.read_csv(r+f'{location3}', sep = " ", header = 0, names = header)
    df3['dataset'] = "FD003"
    df4 = pd.read_csv(r+f'{location4}', sep = " ", header = 0, names = header)
    df4['dataset'] = "FD004"
    return df1, df2, df3, df4

def concatenation(df1, df2, df3, df4):
    train = pd.concat([df1, df2, df3, df4], ignore_index=True)
    train = train.drop(["blank1", "blank2"], axis = 1)
    return train

def remining_cycles(train):
    maxi_df = pd.DataFrame(train.groupby(["esn", "dataset"])["cycles"].max().reset_index())
    maxi_df.rename(columns = {"cycles" : "max_cycles"}, inplace = True)
    train = train.merge(maxi_df, how = "left", on = ["esn", "dataset"])
    train['rem_cycles'] = train['cycles'] - train['max_cycles']
    return train

n_clusters = 6
def clustering(train):
    kmeans = KMeans(n_clusters = n_clusters, 
                    random_state = 0, 
                    n_init="auto").fit(train[["opset1","opset2"]].to_numpy())
    train["condition"] = kmeans.labels_
    return train

lead_time = 100

sensors = ['T2','T24','T30','T50','P2','P15','P30','Nf','Nc','epr','Ps30','phi','NRf','NRc',
            'BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32']
