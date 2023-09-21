import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


    
def outliers_remove(df):
    '''Dealing with Outliers using the IQR method'''
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df
    
def split_data(df):
    '''Split a dataset into train and test sets'''
    # split into input (X) and an output (Y)
    X = df.drop(columns=['status'])
    y = df['status']
     # Call train_test_split with the `stratify` parameter
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42,shuffle=True,stratify=y)
    return X_train, X_test, y_train, y_test