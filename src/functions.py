import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy import stats

def clean_data(data:pd.DataFrame):
    df=data

    general_info(df)
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

    df=remove_duplicates(df)
    check_for_outliers(df)
    print("We WON'T remove the outliers.")

    return df


def general_info(data_df: pd.DataFrame):
    print(f'Shape of dataset: {data_df.shape} ({data_df.shape[0]} entries and {data_df.shape[1]} columns)')
    print(f'Data type of the {data_df.shape[1]} columns\n {data_df.dtypes}')

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
    

def remove_duplicates(data_df: pd.DataFrame):
    """
    We use this function to find and remove any potential duplicates
    """
    
    df=data_df
   
    shape_before=df.shape
    df.drop_duplicates()
    shape_after=df.shape

    if (shape_before[0] != shape_after[0]):
        print("Before removal of duplicates",shape_before)
        print("After removal of duplicates",shape_after)
    else:
        print("No duplicates in the set")
    
    return df

def check_for_outliers(data_df: pd.DataFrame):

    df=data_df
   
    shape_before=df.shape
    
    no_outliers=df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

    shape_after=no_outliers.shape

    if (shape_before[0] != shape_after[0]):
        removed=shape_before[0]-shape_after[0]
        print("Before removal of outliers",shape_before)
        class_imbalance(df)
        print("After removal of outliers",shape_after)
        class_imbalance(no_outliers)
        print(f"We could remove {removed} entries ({(removed/shape_before[0])*100:.2f}% of total entries)")

    else:
        print("No outliers in the set")
    
def class_imbalance(data_df: pd.DataFrame,field='diagnosis'):
    df=data_df
    order=[0,1]

    entries=data_df[field].value_counts()
    print(f'Absolute frequencies of field "{field}"')
    print(entries)

    fractions=data_df[field].value_counts(normalize=True)
    print(f'Percentage of each class of field "{field}"')
    print(fractions)
    