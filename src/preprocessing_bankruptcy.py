import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def median_target(data, var):
    '''Calculate the median values using the target column'''
    temp = data[data[var].notnull()]
    temp = temp[[var, 'class']].groupby(['class'])[[var]].median().reset_index()
    return temp


def replace_median(data, columns):
    ''' Replace NA with median values'''
    for i in columns:
        f = median_target(data, i)
        display(f)
        data.loc[(data['class'] == 0 ) & (data[i].isnull()), i] = f[[i]].values[0][0]
        data.loc[(data['class'] == 1 ) & (data[i].isnull()), i] = f[[i]].values[1][0]
        
    
def outliers_remove(df):
    '''Dealing with Outliers using the IQR method'''
    Q1 = df.quantile(0.02)
    Q3 = df.quantile(0.98)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df
    
def split_data(df):
    '''Split a dataset into train and test sets'''
    # split into input (X) and an output (Y)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1:]
     # Call train_test_split with the `stratify` parameter
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42,shuffle=True,stratify=y)
    return X_train, X_test, y_train, y_test