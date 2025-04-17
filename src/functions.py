import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def clean_data(data:pd.DataFrame):
    df=data

    nan_columns=df.columns[df.isna().any()].tolist()
    total_nan=(df.isna().sum()).sum()

    df_numeric = df.select_dtypes(include=[float, int])
    df_numeric=df_numeric.drop(columns=['id'])

    print(f'Our data consists of {df.shape[1]} columns and {df.shape[0]} entries')
    print(f'We have {df_numeric.shape[1]} numeric columns. These are {list(df_numeric.columns)}')
    print(f'{len(nan_columns)} columns have missing values. These columns are : {nan_columns}')
    print(f'In total we have {total_nan} missing values')

    imp = IterativeImputer(random_state=42)
    df[df_numeric.columns] = imp.fit_transform(df_numeric)

    nan_columns=df.columns[df.isna().any()].tolist()
    print(f'We now have {len(nan_columns)} columns with missing values. These columns are : {nan_columns}')
    
    df['diagnosis']=df['diagnosis'].apply(encode)


    return df


def encode(tumor):
    """
    Method used to encode the entries of the column 'diagnosis'
    Malignant --> 1
    Benign --> 0
    Other --> 2
    """

    if tumor=="M":
        return 0
    elif tumor=="B":
        return 1
    else:
        return 2