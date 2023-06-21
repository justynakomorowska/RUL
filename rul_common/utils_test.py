import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from .config import ConfigTrain as ct

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.float_format',  '{:,}'.format)

header = ['esn','cycles','opset1','opset2','TRA','T2','T24','T30','T50','P2','P15','P30','Nf',
           'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32', "blank1", "blank2"]

header_rul = ['final_rul', 'blank']

test_location1 = r'C:\Users\justyna.komorowska\Documents\___PROJEKTY___\PJATK\Magisterka\data\rawdata\test_FD001.txt'
test_location2 = r'C:\Users\justyna.komorowska\Documents\___PROJEKTY___\PJATK\Magisterka\data\rawdata\test_FD002.txt'
test_location3 = r'C:\Users\justyna.komorowska\Documents\___PROJEKTY___\PJATK\Magisterka\data\rawdata\test_FD003.txt'
test_location4 = r'C:\Users\justyna.komorowska\Documents\___PROJEKTY___\PJATK\Magisterka\data\rawdata\test_FD004.txt'


rul_FD001 = r'C:\Users\justyna.komorowska\Documents\___PROJEKTY___\PJATK\Magisterka\data\rawdata\RUL_FD001.txt'
rul_FD002 = r'C:\Users\justyna.komorowska\Documents\___PROJEKTY___\PJATK\Magisterka\data\rawdata\RUL_FD002.txt'
rul_FD003 = r'C:\Users\justyna.komorowska\Documents\___PROJEKTY___\PJATK\Magisterka\data\rawdata\RUL_FD003.txt'
rul_FD004 = r'C:\Users\justyna.komorowska\Documents\___PROJEKTY___\PJATK\Magisterka\data\rawdata\RUL_FD004.txt'




def read_in_test(file1=test_location1, file2=test_location2, file3=test_location3, file4=test_location4):
    df1 = pd.read_csv(file1, sep = " ", header = 0, names = header)
    df1['dataset'] = "FD001"
    df2 = pd.read_csv(file2, sep = " ", header = 0, names = header)
    df2['dataset'] = "FD002"
    df3 = pd.read_csv(file3, sep = " ", header = 0, names = header)
    df3['dataset'] = "FD003"
    df4 = pd.read_csv(file4, sep = " ", header = 0, names = header)
    df4['dataset'] = "FD004"
    return df1, df2, df3, df4

def read_in_rul(file1=rul_FD001, file2=rul_FD002, file3=rul_FD003, file4=rul_FD004):
    df1 = pd.read_csv(file1, sep = " ", names = header_rul)
    df1['dataset'] = "FD001"
    df1["esn"] = df1.index + 1
    df2 = pd.read_csv(file2, sep = " ",  names = header_rul)
    df2['dataset'] = "FD002"
    df2["esn"] = df2.index + 1
    df3 = pd.read_csv(file3, sep = " ",  names = header_rul)
    df3['dataset'] = "FD003"
    df3["esn"] = df3.index + 1
    df4 = pd.read_csv(file4, sep = " ", names = header_rul)
    df4['dataset'] = "FD004"
    df4["esn"] = df4.index + 1
    return df1, df2, df3, df4



def concatenation(df1, df2, df3, df4):
    """Function working for both: train and test sets - but not for rul set!
    """
    df = pd.concat([df1, df2, df3, df4], ignore_index=True)
    df = df.drop(["blank1", "blank2"], axis = 1)
    return df

def rul_concatenation(df1, df2, df3, df4):
    rul_df = pd.concat([df1, df2, df3, df4], ignore_index=True)
    rul_df.drop(columns="blank", axis=1, inplace = True)
    return rul_df 



def remaining_cycles_test(test, rul_df):
    tmp = pd.DataFrame(test.groupby(["esn", "dataset"])["cycles"].max().reset_index())
    test = test.merge(tmp, how = "inner", on = ["esn", "dataset", "cycles"])
    test = test.merge(rul_df, how = "left", on = ["esn", "dataset"])
    test["rem_cycles"] = test["rem_cycles"] *(-1)
    return test


n_clusters = 6
def clustering(df):
    kmeans = KMeans(n_clusters = n_clusters, 
                    random_state = 0, 
                    n_init="auto").fit(df[["opset1","opset2"]].to_numpy())
    df["condition"] = kmeans.labels_
    return df


lead_time = -100
def make_y(row):
    """lead time is a parameter to be decided with what anticipation the business needs to know 
    about a failure to have enough time for action.
    100 cycles is a convenient anticipation

    This function creates Y_target value if ESN is less that 100 cycles to the Failure or more
    """
    return 1 if row["rem_cycles"] < lead_time else 0

def add_class_col(df):
    df["Class"] = df.apply(make_y, axis = 1)
    return df

def failure_mode(row):
    """Function that creates column with information if there is single HPC Failure Mode (value 0) or HPC + Fan (value 1) FM"""
    return 1 if row['dataset'] in ["FD003", "FD004"] else 0
#test['HPC_Fan'] = test.apply(failure_mode, axis = 1)    

def add_fm(df):
    df["HPC_Fan"] = df.apply(failure_mode, axis = 1)
    return df


def one_hot(df, column, names=None):
    enc = OneHotEncoder()
    labelenc = LabelEncoder()
    if names == None:
        names = df[column].unique().tolist()
    df['column_cat'] = labelenc.fit_transform(df[column])
    enc_df = pd.DataFrame(enc.fit_transform(df[column].to_numpy().reshape(-1,1)).toarray(), columns = names)
    enc_df.drop(columns = names[0], inplace = True)
    df = pd.concat([df, enc_df], axis = 1)
    df.drop(columns = column, inplace = True)
    df.drop(columns = 'column_cat', inplace = True)
    return df


sensors = ['T2','T24','T30','T50','P2','P15','P30','Nf','Nc','epr','Ps30','phi','NRf','NRc',
            'BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32']







predictors = [
'T24',
'T30',
'T50',
'P30',
'Nc',
'epr',
'Ps30',
'phi',
'NRc',
'BPR', #ByPass Ratio
'htBleed',
'W31',
'W32',
'FD002', # dataset
'FD003', # dataset
'FD004', # dataset
'c2', # condition 2
'c3', # condition 3
'c4', # condition 4
'c5', # condition 5
'c6', # condition 6
]

def test_xy_split(df, predictors):
    X_test = df[predictors]
    y_test = df["Class"]
    return X_test, y_test

from sklearn.model_selection import train_test_split


def xy_split(df, predictors, test_size = 0.2, random_state = 42):
    X = df[predictors]
    y = df["Class"]
    X_train, X_val, y_train, y_val = train_test_split(X, 
                                                    y, 
                                                    test_size = test_size, 
                                                    random_state = random_state)

    return X_train, X_val, y_train, y_val